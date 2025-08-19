using Bonsai;
using System;
using System.ComponentModel;
using System.Reactive.Linq;

namespace neurostate_hmm
{
    [Combinator]
    [Description("Rounds a double to the nearest integer. Returns a tuple with the integer value and a boolean indicating if rounding was needed.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class RoundToInteger
    {
        public IObservable<Tuple<int, bool>> Process(IObservable<double> source)
        {
            return source.Select(value =>
            {
                int roundedValue = Convert.ToInt32(value); // Rounds to nearest int
                bool wasInteger = (value == roundedValue); // Checks if rounding was needed

                return Tuple.Create(roundedValue, wasInteger);
            });
        }
    }
}

