using Bonsai;
using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.IO;


namespace neurostate-hmm
{
    [Combinator]
    [Description("Contains all parameters necessary for Preprocessing LFP data, HSMM fitting, and HSMM decoding")]
    [WorkflowElementCategory(ElementCategory.Source)]
    public class InitializeParametersDecoding
    {
        // Define parameter variables

        // Preprocessing parameters
        private int sample_rate;
        private int downsampled_sample_rate;
        private double filter_lower_bound;
        private double filter_upper_bound;
        private double window_duration;
        private double window_step_duration;
        private TimeSpan clock_period_timespan;
        private double low_freq_bound;
        private double high_freq_bound;
        private string parameters_filename;

        // HSMM parameters
        private int num_states;
        private int num_substates;
        private int obs_dim;
        private string train_data_filename;
        private string model_filename;
        private int sequence_length;


        /* The following functions define getters and setters for each variable. When a parameter is entered into the corresponding
        * field for a parameter in Bonsai it is placed in the specified variable. 
        */
        [Description("Sample rate of recording")]
        [Category("Preprocessing")]
        public int SampleRate
        {
            get { return sample_rate; }
            set
            {
                sample_rate = value;
            }
        }


        [Description("Sample rate of recording after downsampling")]
        [Category("Preprocessing")]
        public int DownsampledSampleRate
        {
            get { return downsampled_sample_rate; }
            set
            {
                downsampled_sample_rate = value;
            }
        }

        [Description("Lower bound of bandpass filter")]
        [Category("Preprocessing")]
        public double FilterLowerBound
        {
            get { return filter_lower_bound; }
            set { filter_lower_bound = value; }
        }

        [Description("Upper bound of bandpass filter")]
        [Category("Preprocessing")]
        public double FilterUpperBound
        {
            get { return filter_upper_bound; }
            set { filter_upper_bound = value; }
        }

        [Description("Size of sliding window (s)")]
        [Category("Preprocessing")]
        public double WindowDuration
        {
            get { return window_duration; }
            set
            {
                window_duration = value;
            }
        }

        [Description("Step size of sliding window (s)")]
        [Category("Preprocessing")]
        public double WindowStepDuration
        {
            get { return window_step_duration; }
            set
            {
                window_step_duration = value;
                clock_period_timespan = TimeSpan.FromSeconds(value);
            }
        }

        [Description("Step size of sliding window as TimeSpan object")]
        [Category("Preprocessing")]
        public TimeSpan ClockPeriodTimespan
        {
            get { return clock_period_timespan; }
        }

        [Description("Lower bound of frequencies to consider when inferring state")]
        [Category("Preprocessing")]
        public double LowFrequencyBound
        {
            get { return low_freq_bound; }
            set { low_freq_bound = value; }
        }

        [Description("Upper bound of frequencies to consider when inferring state")]
        [Category("Preprocessing")]
        public double HighFrequencyBound
        {
            get { return high_freq_bound; }
            set { high_freq_bound = value; }
        }

        [Editor(DesignTypes.SaveFileNameEditor, DesignTypes.UITypeEditor)]
        [Description("Path and filename to save parameter details")]
        [Category("Preprocessing")]
        public string Parameters_FN
        {
            get { return parameters_filename; }
            set { parameters_filename = value; }
        }

        [Description("Number of states")]
        [Category("HSMM Fitting Parameters")]
        public int NumStates
        {
            get { return num_states; }
            set { num_states = value; }
        }

        [Description("Number of substates")]
        [Category("HSMM Fitting Parameters")]
        public int NumSubstates
        {
            get { return num_substates; }
            set { num_substates = value; }
        }

        [Description("Observation dimension")]
        [Category("HSMM Fitting Parameters")]
        public int ObsDimension
        {
            get { return obs_dim; }
            set { obs_dim = value; }
        }

        [Editor(DesignTypes.OpenFileNameEditor, DesignTypes.UITypeEditor)]
        [Description("Path and filename of data to train model on")]
        [Category("HSMM Fitting Data")]
        public string Training_Data_FN
        {
            get { return train_data_filename; }
            set { train_data_filename = value; }
        }

        [Editor(DesignTypes.SaveFileNameEditor, DesignTypes.UITypeEditor)]
        [Description("Path and filename of model .pkl file")]
        [Category("HSMM Model")]
        public string ModelFN
        {
            get { return model_filename; }
            set { model_filename = value; }
        }

        [Description("Past sequence length of observations to consider when inferring state")]
        [Category("HSMM Decoding Parameters")]
        public int SequenceLength
        {
            get { return sequence_length; }
            set { sequence_length = value; }
        }

        // Validation logic
        private bool ValidateParameters()
        {
            // Check for integer conditions
            if (!IsInteger(sample_rate * window_duration))
                return false;

            if (!IsInteger(sample_rate * window_step_duration))
                return false;

            if (!IsInteger(downsampled_sample_rate * window_duration))
                return false;

            if (!IsInteger(downsampled_sample_rate * window_step_duration))
                return false;

            if (!IsInteger((float)sample_rate / downsampled_sample_rate))
                return false;

            return true;
        }

        private bool IsInteger(double value)
        {
            return (double)Math.Abs((decimal)value - Math.Round((decimal)value)) < 1e-6;
        }

        private void SaveParametersToFile()
        {
            if (string.IsNullOrEmpty(parameters_filename))
            {
                throw new InvalidOperationException("Parameters filename is not set.");
            }

            using (var writer = new StreamWriter(parameters_filename))
            {
                writer.WriteLine("Preprocessing Parameters:");
                writer.WriteLine("sample_rate: " + sample_rate);
                writer.WriteLine("downsampled_sample_rate: " + downsampled_sample_rate);
                writer.WriteLine("filter_lower_bound: " + filter_lower_bound);
                writer.WriteLine("filter_higher_bound: " + filter_upper_bound);
                writer.WriteLine("window_duration: " + window_duration);
                writer.WriteLine("window_step_duration: " + window_step_duration);
                writer.WriteLine("low_freq_bound: " + low_freq_bound);
                writer.WriteLine("high_freq_bound: " + high_freq_bound);
                writer.WriteLine("Parameters Filename: " + parameters_filename);

                writer.WriteLine("HSMM Parameters:");
                writer.WriteLine("num_states: " + num_states);
                writer.WriteLine("num_substates: " + num_substates);
                writer.WriteLine("obs_dimension: " + obs_dim);
                writer.WriteLine("train_data_filename: " + train_data_filename);
                writer.WriteLine("model_filename: " + model_filename);
                writer.WriteLine("sequence_length: " + sequence_length);
            }
        }

        /*
        * Adds default values for the node parameters
        */
        public InitializeParametersDecoding()
        {}


        /*
        * Returns node parameters
        */
        public IObservable<InitializeParametersDecoding> Process()
        {
            bool valid_params = ValidateParameters();

            if (!valid_params) {
                throw new InvalidOperationException("Parameters are not compatible");
            }

            SaveParametersToFile();
            return Observable.Defer(() => Observable.Return(
                new InitializeParametersDecoding()
                {
                    sample_rate = this.sample_rate,
                    downsampled_sample_rate = this.downsampled_sample_rate,
                    filter_lower_bound = this.filter_lower_bound,
                    filter_upper_bound = this.filter_upper_bound,
                    window_duration = this.window_duration,
                    window_step_duration = this.window_step_duration,
                    clock_period_timespan = this.clock_period_timespan,
                    low_freq_bound = this.low_freq_bound,
                    high_freq_bound = this.high_freq_bound,
                    parameters_filename = this.parameters_filename,
                    num_states = this.num_states,
                    num_substates = this.num_substates,
                    obs_dim = this.obs_dim,
                    train_data_filename = this.train_data_filename,
                    model_filename = this.model_filename,
                    sequence_length = this.sequence_length
                }
            ));
            
        }

    }
}