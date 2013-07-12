import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Set;

import com.aliasi.classify.Classified;
import com.aliasi.classify.ConditionalClassifier;
import com.aliasi.classify.ConditionalClassifierEvaluator;
import com.aliasi.classify.ConfusionMatrix;
import com.aliasi.classify.PrecisionRecallEvaluation;
import com.aliasi.classify.TradNaiveBayesClassifier;
import com.aliasi.corpus.Corpus;
import com.aliasi.corpus.ObjectHandler;
import com.aliasi.tokenizer.RegExTokenizerFactory;
import com.aliasi.tokenizer.TokenizerFactory;
import com.aliasi.util.AbstractExternalizable;
import com.aliasi.util.CollectionUtils;

public class ClassifyWithNBC {

	public static void go(String datapath) throws IOException,
			ClassNotFoundException {
		File corpusFile = new File(datapath);

		Corpus<ObjectHandler<Classified<CharSequence>>> corpus = new LabeledAbstractCorpus(
				corpusFile);

		String[] CATEGORIES = ((LabeledAbstractCorpus) corpus).getCatogories();
		// String TRAINING_DIR = "./data";
		TokenizerFactory tf = new RegExTokenizerFactory("\\P{Z}+");
		// Set<String> cats = CollectionUtils.asSet(CATEGORIES);
		double catPrior = 1.0;
		double tokPrior = 1;
		double lengthNorm = Double.NaN;
		Set<String> catSet = CollectionUtils.asSet(CATEGORIES);
		// String[] cats = catSet.toArray(Strings.EMPTY_STRING_ARRAY);
		// Arrays.sort(cats);

		TradNaiveBayesClassifier classifier = new TradNaiveBayesClassifier(
				catSet, tf, catPrior, tokPrior, lengthNorm);

		corpus.visitTrain(classifier);

		// System.out.println("COMPILING");
		@SuppressWarnings("unchecked")
		ConditionalClassifier<CharSequence> cc = (ConditionalClassifier<CharSequence>) AbstractExternalizable
				.compile(classifier);

		System.out.println("EVALUATING");
		boolean storeInputs = false;
		ConditionalClassifierEvaluator<CharSequence> evaluator = new ConditionalClassifierEvaluator<CharSequence>(
				cc, CATEGORIES, storeInputs);
		corpus.visitTest(evaluator);

		ConfusionMatrix confMatrix = evaluator.confusionMatrix();

		System.out.println("Total Accuracy: " + confMatrix.totalAccuracy());

		ConfusionMatrix cm = evaluator.confusionMatrix();
		// int totalCount = cm.totalCount();
		// int totalCorrect = cm.totalCorrect();
		// double accuracy = cm.totalAccuracy();
		// System.out.println("Macro evaluation:");
		// double macroAvgPrec = cm.macroAvgPrecision();
		// double macroAvgRec = cm.macroAvgRecall();
		// double macroAvgF = cm.macroAvgFMeasure();
		// // double kappa = cm.kappa();
		// /* x */
		// System.out.println("\nAll Versus All");
		// System.out.println("  correct/total = " + totalCorrect + " / "
		// + totalCount);
		// System.out.printf("  accuracy=%5.3f\n", accuracy);
		// System.out.printf("  Macro Avg: prec=%5.3f  rec=%5.3f   F=%5.3f\n",
		// macroAvgPrec, macroAvgRec, macroAvgF);

		PrecisionRecallEvaluation pre = cm.microAverage();
		System.out.println("Average one-vs-all confusion matrix");
		System.out.println("============");
		System.out.printf("  tp=%d  fn=%d\n  fp=%d  tn=%d\n",
				pre.truePositive(), pre.falseNegative(), pre.falsePositive(),
				pre.trueNegative());
		System.out.println("============");
		System.out.printf("  accuracy=%5.3f\n", pre.accuracy());
		System.out.printf("  Micro Avg: prec=%5.3f  rec=%5.3f   F=%5.3f\n",
				pre.precision(), pre.recall(), pre.fMeasure());

		// System.out.printf("  Kappa=%5.3f\n", kappa);

		for (int k = 0; k < CATEGORIES.length; ++k) {
			PrecisionRecallEvaluation pr = confMatrix.oneVsAll(k);
			long tp = pr.truePositive();
			long tn = pr.trueNegative();
			long fp = pr.falsePositive();
			long fn = pr.falseNegative();

			double acc = pr.accuracy();

			double prec = pr.precision();
			double recall = pr.recall();
			double specificity = pr.rejectionRecall();
			double f = pr.fMeasure();

			System.out.println("\n*Category[" + k + "]=" + CATEGORIES[k]
					+ " versus All");
			System.out.println("  * TP=" + tp + " TN=" + tn + " FP=" + fp
					+ " FN=" + fn);
			System.out.printf("  * Accuracy=%5.3f\n", acc);
			System.out.printf(
					"  * Prec=%5.3f  Rec(Sens)=%5.3f  Spec=%5.3f  F=%5.3f\n",
					prec, recall, specificity, f);

		}
	}

	public static void main(String[] args) throws IOException,
			ClassNotFoundException {

		// Train a initial Naive Bayesian classifier
		// read from memory
		// Set<String> cats = CollectionUtils.asSet("hers", "his");
		// TokenizerFactory tf = new RegExTokenizerFactory("\\P{Z}+");
		// double catPrior = 1.0;
		// double tokenPrior = 0.5;
		// double lengthNorm = Double.NaN;
		// TradNaiveBayesClassifier initClassifier = new
		// TradNaiveBayesClassifier(
		// cats, tf);
		//
		// Classification hersCl = new Classification("hers");
		//
		// List<String> herTexts = Arrays.asList("haw hee", "hee hee hee haw",
		// "haw");
		// for (String t : herTexts) {
		// initClassifier.handle(new Classified<CharSequence>(t, hersCl));
		// }
		//
		// String testText = "haw hee haw";
		//
		// JointClassification jc = initClassifier.classify(testText);
		// for (int rank = 0; rank < jc.size(); rank++) {
		// String cat = jc.category(rank);
		// double condProb = jc.conditionalProbability(rank);
		// double jointProb = jc.jointLog2Probability(rank);
		// System.out.println(cat);
		// System.out.println(condProb);
		// System.out.println(jointProb);
		// }

		// //////////////////////////////
		// read from disk
		// String[] CATEGORIES = { "hers", "his" };
		// String TRAINING_DIR = "./data";
		// TokenizerFactory tf = new RegExTokenizerFactory("\\P{Z}+");
		// Set<String> cats = CollectionUtils.asSet(CATEGORIES);
		// TradNaiveBayesClassifier classifier = new TradNaiveBayesClassifier(
		// cats, tf);
		//
		// for (int i = 0; i < CATEGORIES.length; i++) {
		// File classDir = new File(TRAINING_DIR, CATEGORIES[i]);
		// if (!classDir.isDirectory()) {
		// String msg = "Could not find training directory=" + classDir
		// + "\nHave you unpacked 4 newsgroups?";
		// System.out.println(msg); // in case exception gets lost in shell
		// throw new IllegalArgumentException(msg);
		// }
		//
		// String[] trainingFiles = classDir.list();
		// for (int j = 0; j < trainingFiles.length; ++j) {
		// File file = new File(classDir, trainingFiles[j]);
		// String text = Files.readFromFile(file, "utf-8");
		// System.out.println("Training on " + CATEGORIES[i] + "/"
		// + trainingFiles[j]);
		// Classification classification = new Classification(
		// CATEGORIES[i]);
		// Classified<CharSequence> classified = new Classified<CharSequence>(
		// text, classification);
		// classifier.handle(classified);
		// }
		//
		// }
		//
		// String testText = "hee hee";
		//
		// JointClassification jc = classifier.classify(testText);
		// for (int rank = 0; rank < jc.size(); rank++) {
		// String cat = jc.category(rank);
		// double condProb = jc.conditionalProbability(rank);
		// // double jointProb = jc.jointLog2Probability(rank);
		// System.out.println(cat);
		// System.out.println(condProb);
		// // System.out.println(jointProb);
		// }

		// /////////////////
		// read from corpus

		try {
			System.setOut(new PrintStream(new File("detail_result.txt")));
		} catch (Exception e) {
			e.printStackTrace();
		}

		//
		// data_root_directory
		// |---- part directory (1-9, 2-8, 3-7 ...)
		// |-----|--- percent directory (0.1, 0.2, 0.3 ... 1.0)
		// |-----|----|--- feature type directory (ig, mi, ti)
		File dataDir = new File("C:\\cygwin\\home\\Mark\\tokenization.new");
		File[] partDirs = dataDir.listFiles();
		for (int i = 0; i < partDirs.length; i++) {
			File part = partDirs[i];

			File[] featTypeDirs = part.listFiles();
			for (int j = 0; j < featTypeDirs.length; j++) {
				File typeDir = featTypeDirs[j];

				if (typeDir.isDirectory()) {
					File[] percentDirs = typeDir.listFiles();
					for (int k = 0; k < percentDirs.length; k++) {
						System.out.println("\n\n"
								+ percentDirs[k].getAbsolutePath());
						go(percentDirs[k].getAbsolutePath());
					}
				}
			}
		}

		// System.out.println("\n\n\n========== 2-8 =============");
		// go(".\\data\\abstract\\2-8");
		// System.out.println("\n\n\n========== 3-7 =============");
		// go(".\\data\\abstract\\3-7");
		// System.out.println("\n\n\n========== 4-6 =============");
		// go(".\\data\\abstract\\4-6");
		// System.out.println("\n\n\n========== 5-5 =============");
		// go(".\\data\\abstract\\5-5");

		// System.out.println("\n======FULL EVAL======");
		// System.out.println(evaluator);

		// ////////
		// EM naive bayes
		// int MAX_ITER = 100;
		// double minTokenCount = 1;
		// double minImprovement = 1;
		// TradNaiveBayesClassifier lastClassifier = TradNaiveBayesClassifier
		// .emTrain(classifier, arg1, arg2, arg3, minTokenCount, MAX_ITER,
		// minImprovement, null/* no reporter */);

	}
}
