import java.io.IOException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class MaxTemperature {

    public static class TempMapper extends Mapper<LongWritable, Text, Text, IntWritable> {
        public void map(LongWritable key, Text value, Context ctx) throws IOException, InterruptedException {
            String[] f = value.toString().trim().split("\\s+");
            if (f.length >= 2)
                ctx.write(new Text(f[0]), new IntWritable(Integer.parseInt(f[1])));
        }
    }

    public static class TempReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        public void reduce(Text key, Iterable<IntWritable> vals, Context ctx) throws IOException, InterruptedException {
            int max = Integer.MIN_VALUE;
            for (IntWritable v : vals) max = Math.max(max, v.get());
            ctx.write(key, new IntWritable(max));
        }
    }

    public static void main(String[] args) throws Exception {
        Job job = Job.getInstance(new Configuration(), "Max Temperature");
        job.setJarByClass(MaxTemperature.class);
        job.setMapperClass(TempMapper.class);
        job.setCombinerClass(TempReducer.class);
        job.setReducerClass(TempReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
