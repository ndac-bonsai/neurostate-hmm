using Bonsai;
using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;

namespace neurostate_hmm
{
    [Combinator]
    [Description("Zips an arbitrary number of input sequences into a single sequence of tuples.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class ZipN
    {
        public IObservable<object[]> Process(params IObservable<object>[] sources)
        {
            if (sources == null || sources.Length == 0)
            {
                throw new ArgumentException("At least one input sequence is required.");
            }

            return Observable.Zip(sources).Select(values => values.ToArray());
        }
    }
}
