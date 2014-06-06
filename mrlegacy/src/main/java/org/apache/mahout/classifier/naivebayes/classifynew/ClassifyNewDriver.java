/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.classifier.naivebayes.classifynew;

import org.apache.mahout.classifier.naivebayes.test.*;
import com.google.common.base.Preconditions;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

import com.google.common.io.Closeables;
import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.mahout.classifier.ClassifierResult;
import org.apache.mahout.classifier.ResultAnalyzer;
import org.apache.mahout.classifier.naivebayes.AbstractNaiveBayesClassifier;
import org.apache.mahout.classifier.naivebayes.BayesUtils;
import org.apache.mahout.classifier.naivebayes.ComplementaryNaiveBayesClassifier;
import org.apache.mahout.classifier.naivebayes.NaiveBayesModel;
import org.apache.mahout.classifier.naivebayes.StandardNaiveBayesClassifier;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;
import org.apache.mahout.common.lucene.AnalyzerUtils;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.vectorizer.collocations.llr.LLRReducer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Classifys a new text document by tokeninzing and calculating 
 * new a term frequency based on the dictonary used in training
 * terms not in the dictionary will be ignored and passing them 
 * to a previously trained (Complementary) Naive Bayes model.
 * 
 * In the case of that the model was trained using TF-IDF vectors, 
 * the IDF will be calculated using the Document Frequencies 
 * respective to the dictionary.
 * 
 * The model must have been trained, and the documents vectorized 
 * using the same parameters as provided here.
 *  
 */

public class ClassifyNewDriver extends AbstractJob {

  private static final Logger log = LoggerFactory.getLogger(ClassifyNewDriver.class);

  public static final String COMPLEMENTARY = "class"; //b for bayes, c for complementary
  private static final Pattern SLASH = Pattern.compile("/");
 
  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new ClassifyNewDriver(), args);
  }

  @Override
  public int run(String[] args) throws Exception {
    addInputOption();
    addOutputOption();
    
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();


    // Naive Bayes Model options
    addOption(addOption(DefaultOptionCreator.overwriteOption().create()));
    addOption("model", "m", "The path to the model built during training", 
            true);
    addOption("dictionary", "d", "The path to dictionary used for training", 
            true);
    addOption("dfVectors", "dv", "The path to the document frequency vectors "
            + "used fortraining", true);   
    addOption(buildOption("testComplementary", "c", "test complementary?", 
            false, false, String.valueOf(false)));
    addOption(buildOption("runSequential", "seq", "run sequential?",
            false, false, String.valueOf(false)));
    addOption("labelIndex", "l", "The path to the location of the label index",
            true);
    Map<String, List<String>> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }
    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      HadoopUtil.delete(getConf(), getOutputPath());
    }


   // Dictionary and Vectorization options
   Option inputDirOpt = DefaultOptionCreator.inputOption().create();

   Option outputDirOpt = DefaultOptionCreator.outputOption().create();


   Option minSupportOpt = obuilder.withLongName("minSupport").withArgument(
            abuilder.withName("minSupport").withMinimum(1).withMaximum(1).create()).withDescription(
            "(Optional) Minimum Support. Default Value: 2").withShortName("s").create();

    Option analyzerNameOpt = obuilder.withLongName("analyzerName").withArgument(
            abuilder.withName("analyzerName").withMinimum(1).withMaximum(1).create()).withDescription(
            "The class name of the analyzer").withShortName("a").create();

    Option chunkSizeOpt = obuilder.withLongName("chunkSize").withArgument(
            abuilder.withName("chunkSize").withMinimum(1).withMaximum(1).create()).withDescription(
            "The chunkSize in MegaBytes. Default Value: 100MB").withShortName("chunk").create();

    Option weightOpt = obuilder.withLongName("weight").withRequired(false).withArgument(
            abuilder.withName("weight").withMinimum(1).withMaximum(1).create()).withDescription(
            "The kind of weight to use. Currently TF or TFIDF. Default: TFIDF").withShortName("wt").create();

    Option minDFOpt = obuilder.withLongName("minDF").withRequired(false).withArgument(
            abuilder.withName("minDF").withMinimum(1).withMaximum(1).create()).withDescription(
            "The minimum document frequency.  Default is 1").withShortName("md").create();

    Option maxDFPercentOpt = obuilder.withLongName("maxDFPercent").withRequired(false).withArgument(
            abuilder.withName("maxDFPercent").withMinimum(1).withMaximum(1).create()).withDescription(
            "The max percentage of docs for the DF.  Can be used to remove really high frequency terms."
                    + " Expressed as an integer between 0 and 100. Default is 99.  If maxDFSigma is also set, "
                    + "it will override this value.").withShortName("x").create();

    Option maxDFSigmaOpt = obuilder.withLongName("maxDFSigma").withRequired(false).withArgument(
            abuilder.withName("maxDFSigma").withMinimum(1).withMaximum(1).create()).withDescription(
            "What portion of the tf (tf-idf) vectors to be used, expressed in times the standard deviation (sigma) "
                    + "of the document frequencies of these vectors. Can be used to remove really high frequency terms."
                    + " Expressed as a double value. Good value to be specified is 3.0. In case the value is less "
                    + "than 0 no vectors will be filtered out. Default is -1.0.  Overrides maxDFPercent")
            .withShortName("xs").create();

    Option minLLROpt = obuilder.withLongName("minLLR").withRequired(false).withArgument(
            abuilder.withName("minLLR").withMinimum(1).withMaximum(1).create()).withDescription(
            "(Optional)The minimum Log Likelihood Ratio(Float)  Default is " + LLRReducer.DEFAULT_MIN_LLR)
            .withShortName("ml").create();

    Option numReduceTasksOpt = obuilder.withLongName("numReducers").withArgument(
            abuilder.withName("numReducers").withMinimum(1).withMaximum(1).create()).withDescription(
            "(Optional) Number of reduce tasks. Default Value: 1").withShortName("nr").create();

    Option powerOpt = obuilder.withLongName("norm").withRequired(false).withArgument(
            abuilder.withName("norm").withMinimum(1).withMaximum(1).create()).withDescription(
            "The norm to use, expressed as either a float or \"INF\" if you want to use the Infinite norm.  "
                    + "Must be greater or equal to 0.  The default is not to normalize").withShortName("n").create();

    Option logNormalizeOpt = obuilder.withLongName("logNormalize").withRequired(false)
            .withDescription(
                    "(Optional) Whether output vectors should be logNormalize. If set true else false")
            .withShortName("lnorm").create();

    Option maxNGramSizeOpt = obuilder.withLongName("maxNGramSize").withRequired(false).withArgument(
            abuilder.withName("ngramSize").withMinimum(1).withMaximum(1).create())
            .withDescription(
                    "(Optional) The maximum size of ngrams to create"
                            + " (2 = bigrams, 3 = trigrams, etc) Default Value:1").withShortName("ng").create();

    Option sequentialAccessVectorOpt = obuilder.withLongName("sequentialAccessVector").withRequired(false)
            .withDescription(
                    "(Optional) Whether output vectors should be SequentialAccessVectors. If set true else false")
            .withShortName("seq").create();

    Option namedVectorOpt = obuilder.withLongName("namedVector").withRequired(false)
            .withDescription(
                    "(Optional) Whether output vectors should be NamedVectors. If set true else false")
            .withShortName("nv").create();

    Option overwriteOutput = obuilder.withLongName("overwrite").withRequired(false).withDescription(
            "If set, overwrite the output directory").withShortName("ow").create();
    Option helpOpt = obuilder.withLongName("help").withDescription("Print out help").withShortName("h")
            .create();

    Group group = gbuilder.withName("Options").withOption(minSupportOpt).withOption(analyzerNameOpt)
            .withOption(chunkSizeOpt).withOption(outputDirOpt).withOption(inputDirOpt).withOption(minDFOpt)
            .withOption(maxDFSigmaOpt).withOption(maxDFPercentOpt).withOption(weightOpt).withOption(powerOpt)
            .withOption(minLLROpt).withOption(numReduceTasksOpt).withOption(maxNGramSizeOpt).withOption(overwriteOutput)
            .withOption(helpOpt).withOption(sequentialAccessVectorOpt).withOption(namedVectorOpt)
            .withOption(logNormalizeOpt)
            .create();
    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      parser.setHelpOption(helpOpt);
      CommandLine cmdLine = parser.parse(args);

      if (cmdLine.hasOption(helpOpt)) {
        CommandLineUtil.printHelp(group);
        return -1;
      }

      Path inputDir = new Path((String) cmdLine.getValue(inputDirOpt));
      Path outputDir = new Path((String) cmdLine.getValue(outputDirOpt));

      int chunkSize = 100;
      if (cmdLine.hasOption(chunkSizeOpt)) {
        chunkSize = Integer.parseInt((String) cmdLine.getValue(chunkSizeOpt));
      }
      int minSupport = 2;
      if (cmdLine.hasOption(minSupportOpt)) {
        String minSupportString = (String) cmdLine.getValue(minSupportOpt);
        minSupport = Integer.parseInt(minSupportString);
      }

      int maxNGramSize = 1;

      if (cmdLine.hasOption(maxNGramSizeOpt)) {
        try {
          maxNGramSize = Integer.parseInt(cmdLine.getValue(maxNGramSizeOpt).toString());
        } catch (NumberFormatException ex) {
          log.warn("Could not parse ngram size option");
        }
      }
      log.info("Maximum n-gram size is: {}", maxNGramSize);

      if (cmdLine.hasOption(overwriteOutput)) {
        HadoopUtil.delete(getConf(), outputDir);
      }

      float minLLRValue = LLRReducer.DEFAULT_MIN_LLR;
      if (cmdLine.hasOption(minLLROpt)) {
        minLLRValue = Float.parseFloat(cmdLine.getValue(minLLROpt).toString());
      }
      log.info("Minimum LLR value: {}", minLLRValue);

      int reduceTasks = 1;
      if (cmdLine.hasOption(numReduceTasksOpt)) {
        reduceTasks = Integer.parseInt(cmdLine.getValue(numReduceTasksOpt).toString());
      }
      log.info("Number of reduce tasks: {}", reduceTasks);

      Class<? extends Analyzer> analyzerClass = StandardAnalyzer.class;
      if (cmdLine.hasOption(analyzerNameOpt)) {
        String className = cmdLine.getValue(analyzerNameOpt).toString();
        analyzerClass = Class.forName(className).asSubclass(Analyzer.class);
        // try instantiating it, b/c there isn't any point in setting it if
        // you can't instantiate it
        AnalyzerUtils.createAnalyzer(analyzerClass);
      }

      boolean processIdf;

      if (cmdLine.hasOption(weightOpt)) {
        String wString = cmdLine.getValue(weightOpt).toString();
        if ("tf".equalsIgnoreCase(wString)) {
          processIdf = false;
        } else if ("tfidf".equalsIgnoreCase(wString)) {
          processIdf = true;
        } else {
          throw new OptionException(weightOpt);
        }
      } else {
        processIdf = true;
      }

      int minDf = 1;
      if (cmdLine.hasOption(minDFOpt)) {
        minDf = Integer.parseInt(cmdLine.getValue(minDFOpt).toString());
      }
      int maxDFPercent = 99;
      if (cmdLine.hasOption(maxDFPercentOpt)) {
        maxDFPercent = Integer.parseInt(cmdLine.getValue(maxDFPercentOpt).toString());
      }
      double maxDFSigma = -1.0;
      if (cmdLine.hasOption(maxDFSigmaOpt)) {
        maxDFSigma = Double.parseDouble(cmdLine.getValue(maxDFSigmaOpt).toString());
      }
      
      
      
      // TODO: 1. Check if Dictionary is pruned down using all above 
      // options before reading them in again.
      // 1113827 in the dictionary
      // 1113828 in the dfcount
      
      
      
    }catch(Exception e){
      e.printStackTrace();
    }
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    // Classify with the NB Model

    boolean sequential = hasOption("runSequential");
    boolean succeeded;
    if (sequential) {
       runSequential();
    } else {
      succeeded = runMapReduce();
      if (!succeeded) {
        return -1;
      }
    }

    //load the labels
    Map<Integer, String> labelMap = BayesUtils.readLabelIndex(getConf(), new Path(getOption("labelIndex")));

    //loop over the results and create the confusion matrix
    SequenceFileDirIterable<Text, VectorWritable> dirIterable =
        new SequenceFileDirIterable<Text, VectorWritable>(getOutputPath(), PathType.LIST, PathFilters.partFilter(), getConf());
    ResultAnalyzer analyzer = new ResultAnalyzer(labelMap.values(), "DEFAULT");
    analyzeResults(labelMap, dirIterable, analyzer);

    log.info("{} Results: {}", hasOption("testComplementary") ? "Complementary" : "Standard NB", analyzer);
    return 0;
  }

  private void runSequential() throws IOException {
    boolean complementary = hasOption("testComplementary");
    FileSystem fs = FileSystem.get(getConf());
    NaiveBayesModel model = NaiveBayesModel.materialize(new Path(getOption("model")), getConf());
    
    // Ensure that if we are testing in complementary mode, the model has been
    // trained complementary. a complementarty model will work for standard classification
    // a standard model will not work for complementary classification
    if (complementary){
        Preconditions.checkArgument((model.isComplemtary() == complementary),
            "Complementary mode in model is different from test mode");
    }
    
    AbstractNaiveBayesClassifier classifier;
    if (complementary) {
      classifier = new ComplementaryNaiveBayesClassifier(model);
    } else {
      classifier = new StandardNaiveBayesClassifier(model);
    }
    SequenceFile.Writer writer = SequenceFile.createWriter(fs, getConf(), new Path(getOutputPath(), "part-r-00000"),
        Text.class, VectorWritable.class);

    try {
      SequenceFileDirIterable<Text, VectorWritable> dirIterable =
          new SequenceFileDirIterable<Text, VectorWritable>(getInputPath(), PathType.LIST, PathFilters.partFilter(), getConf());
      // loop through the part-r-* files in getInputPath() and get classification scores for all entries
      for (Pair<Text, VectorWritable> pair : dirIterable) {
        writer.append(new Text(SLASH.split(pair.getFirst().toString())[1]),
            new VectorWritable(classifier.classifyFull(pair.getSecond().get())));
      }
    } finally {
      Closeables.close(writer, false);
    }
  }

  private boolean runMapReduce() throws IOException,
      InterruptedException, ClassNotFoundException {
    Path model = new Path(getOption("model"));
    HadoopUtil.cacheFiles(model, getConf());
    //the output key is the expected value, the output value are the scores for all the labels
    Job testJob = prepareJob(getInputPath(), getOutputPath(), SequenceFileInputFormat.class, BayesTestMapper.class,
        Text.class, VectorWritable.class, SequenceFileOutputFormat.class);
    //testJob.getConfiguration().set(LABEL_KEY, getOption("--labels"));


    boolean complementary = hasOption("testComplementary");
    testJob.getConfiguration().set(COMPLEMENTARY, String.valueOf(complementary));
    return testJob.waitForCompletion(true);
  }

  private static void analyzeResults(Map<Integer, String> labelMap,
                                     SequenceFileDirIterable<Text, VectorWritable> dirIterable,
                                     ResultAnalyzer analyzer) {
    for (Pair<Text, VectorWritable> pair : dirIterable) {
      int bestIdx = Integer.MIN_VALUE;
      double bestScore = Long.MIN_VALUE;
      for (Vector.Element element : pair.getSecond().get().all()) {
        if (element.get() > bestScore) {
          bestScore = element.get();
          bestIdx = element.index();
        }
      }
      if (bestIdx != Integer.MIN_VALUE) {
        ClassifierResult classifierResult = new ClassifierResult(labelMap.get(bestIdx), bestScore);
        analyzer.addInstance(pair.getFirst().toString(), classifierResult);
      }
    }
  }
}
