using Bonsai;
using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using OpenCV.Net;

namespace neurostate_hmm
{
    [Combinator]
    [Description("Creates an observable sequence of an OpenCV Mat.")]
    [WorkflowElementCategory(ElementCategory.Source)]
    public class OpenCVMat
    {
        [Description("Mat Length.")]
        public int length { get; set; }

        public IObservable<Mat> Process(IObservable<double> source)
        {
            return Observable.Defer(() =>
            {
                return source.Select(value =>
                {
                    double[] doubles = new double[length];
                    for (int j = 0; j < doubles.Length; j++)
                    {
                        doubles[j] = value;
                    }
                    return OpenCV.Net.Mat.FromArray(doubles);
                });
            });
        }
    }
}

