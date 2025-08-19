using Bonsai;
using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;

namespace neurostate_hmm
{
    [Combinator]
    [Description("Takes the log base 10 of each element in every incoming float array. Returns a float array")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class log10
    {
        public IObservable<float[]> Process<T>(IObservable<T[]> source) where T : struct, IConvertible
        {
            return source.Select(value =>
            {
                return value.Select(v => (float)Math.Log10(Convert.ToDouble(v))).ToArray();
            });
        }
    }
}
