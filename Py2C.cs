using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using Python.Runtime;

namespace neurostate_hmm
{
    [Combinator]
    [Description("Converts an incoming stream of Python objects of numpy float64 arrays to C# double arrays (F64)")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class Py2C
    {
        public IObservable<double[]> Process(IObservable<PyObject> source)
        {
            return source.Select(pythonObject =>
            {
                // Create a list to store all doubles from the Python numpy array
                List<double> list = new List<double>();

                // Begin Python Global Interpreter Lock
                using (Py.GIL())
                {
                    // Ensure the Python object is iterable (i.e., a NumPy array)
                    if (pythonObject.IsIterable())
                    {
                        var iterator = pythonObject.GetIterator();
                        while (iterator.MoveNext())
                        {
                            var prob = iterator.Current;
                            try
                            {
                                double value = prob.As<double>(); // Convert to double
                                list.Add(value);
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine("Error converting item to double: " + ex.Message);
                                return new double[0]; // Return an empty array on failure
                            }
                            finally
                            {
                                prob.Dispose();
                            }
                        }
                    }
                    else
                    {
                        Console.WriteLine("Object is not iterable");
                        return new double[0]; // Return an empty array if not iterable
                    }
                }

                // Convert the C# list to a double array and return
                return list.ToArray();
            });
        }
    }
}
