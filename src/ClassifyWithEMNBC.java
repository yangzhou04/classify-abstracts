import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Set;

import com.aliasi.classify.ConditionalClassifier;
import com.aliasi.classify.ConditionalClassifierEvaluator;
import com.aliasi.classify.ConfusionMatrix;
import com.aliasi.classify.PrecisionRecallEvaluation;
import com.aliasi.classify.TradNaiveBayesClassifier;
import com.aliasi.io.LogLevel;
import com.aliasi.io.Reporter;
import com.aliasi.io.Reporters;
import com.aliasi.tokenizer.RegExTokenizerFactory;
import com.aliasi.tokenizer.TokenizerFactory;
import com.aliasi.util.AbstractExternalizable;
import com.aliasi.util.CollectionUtils;
import com.aliasi.util.Factory;

public class ClassifyWithEMNBC {

	public static void go(String datapath) throws IOException,
			ClassNotFoundException {
		File corpusFile = new File(datapath);

		LabeledAbstractCorpus labeledCorpus = new LabeledAbstractCorpus(
				corpusFile);

		// train initial classifier
		final String[] CATEGORIES = labeledCorpus.getCatogories();
		TokenizerFactory tf = new RegExTokenizerFactory("\\P{Z}+");
		double catPrior = 1.0;
		double tokPrior = 1;
		double lengthNorm = Double.NaN;
		Set<String> catSet = CollectionUtils.asSet(CATEGORIES);
		TradNaiveBayesClassifier initClassifier = new TradNaiveBayesClassifier(
				catSet, tf, catPrior, tokPrior, lengthNorm);
		labeledCorpus.visitTrain(initClassifier);

		// read unlabeled corpus
		UnlabeledAbstractCorpus unlabeledCorpus = new UnlabeledAbstractCorpus(
				"C:\\cygwin\\home\\Mark\\unlabeledCorpus.txt");

		// EM naive bayes
		int MAX_ITER = 100;
		double minTokenCount = 2;
		double minImprovement = 1;

		Factory<TradNaiveBayesClassifier> nbcFactory = new Factory<TradNaiveBayesClassifier>() {
			@Override
			public TradNaiveBayesClassifier create() {
				TokenizerFactory tf = new RegExTokenizerFactory("\\P{Z}+");
				double catPrior = 1.0;
				double tokPrior = 1;
				double lengthNorm = Double.NaN;
				Set<String> catSet = CollectionUtils.asSet(CATEGORIES);
				TradNaiveBayesClassifier classifier = new TradNaiveBayesClassifier(
						catSet, tf, catPrior, tokPrior, lengthNorm);
				return classifier;
			}
		};

		Reporter reporter = Reporters.stream(System.out, "utf-8").setLevel(
				LogLevel.DEBUG);

		// using em to train naive bayes classifier
		TradNaiveBayesClassifier emClassifier = TradNaiveBayesClassifier
				.emTrain(initClassifier, nbcFactory, labeledCorpus,
						unlabeledCorpus, minTokenCount, MAX_ITER,
						minImprovement, reporter);

		@SuppressWarnings("unchecked")
		ConditionalClassifier<CharSequence> cc = (ConditionalClassifier<CharSequence>) AbstractExternalizable
				.compile(emClassifier);

		System.out.println("EVALUATING");
		boolean storeInputs = false;
		ConditionalClassifierEvaluator<CharSequence> evaluator = new ConditionalClassifierEvaluator<CharSequence>(
				cc, CATEGORIES, storeInputs);
		labeledCorpus.visitTest(evaluator);

		ConfusionMatrix confMatrix = evaluator.confusionMatrix();

		System.out.println("Total Accuracy: " + confMatrix.totalAccuracy());

		ConfusionMatrix cm = evaluator.confusionMatrix();
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

//		for (int k = 0; k < CATEGORIES.length; ++k) {
//			PrecisionRecallEvaluation pr = confMatrix.oneVsAll(k);
//			long tp = pr.truePositive();
//			long tn = pr.trueNegative();
//			long fp = pr.falsePositive();
//			long fn = pr.falseNegative();
//
//			double acc = pr.accuracy();
//
//			double prec = pr.precision();
//			double recall = pr.recall();
//			double specificity = pr.rejectionRecall();
//			double f = pr.fMeasure();
//
//			System.out.println("\n*Category[" + k + "]=" + CATEGORIES[k]
//					+ " versus All");
//			System.out.println("  * TP=" + tp + " TN=" + tn + " FP=" + fp
//					+ " FN=" + fn);
//			System.out.printf("  * Accuracy=%5.3f\n", acc);
//			System.out.printf(
//					"  * Prec=%5.3f  Rec(Sens)=%5.3f  Spec=%5.3f  F=%5.3f\n",
//					prec, recall, specificity, f);
//		}
	}

	public static void main(String[] args) throws IOException,
			ClassNotFoundException {

		try {
			System.setOut(new PrintStream(new File("micro_em_result.txt")));
		} catch (Exception e) {
			e.printStackTrace();
		}
		// read labeled corpus
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

		System.out.println("Done!");
	}
}
