import java.io.File;
import java.io.IOException;
import java.util.StringTokenizer;

import com.aliasi.classify.Classification;
import com.aliasi.classify.Classified;
import com.aliasi.corpus.Corpus;
import com.aliasi.corpus.ObjectHandler;
import com.aliasi.util.Files;

public class LabeledAbstractCorpus extends
		Corpus<ObjectHandler<Classified<CharSequence>>> {

	private final File mCorpusFile;

	private String[] mCats;
	private String mTrainDir;
	private String mTestDir;

	public LabeledAbstractCorpus(File corpusFile) throws IOException {
		mCorpusFile = corpusFile;
		mTrainDir = mCorpusFile.getAbsolutePath() + "\\train";
		mTestDir = mCorpusFile.getAbsolutePath() + "\\test";
	}

	@Override
	public void visitTrain(ObjectHandler<Classified<CharSequence>> handler)
			throws IOException {

		if (mCats == null)
			mCats = getCatogories();

		for (int i = 0; i < mCats.length; i++) {
			File classDir = new File(mTrainDir, mCats[i]);
			if (!classDir.isDirectory()) {
				String msg = "Could not find training directory=" + classDir;
				System.out.println(msg); // in case exception gets lost in shell
				throw new IllegalArgumentException(msg);
			}

			String[] trainingFiles = classDir.list();
			for (int j = 0; j < trainingFiles.length; ++j) {
				File file = new File(classDir, trainingFiles[j]);
				String text = Files.readFromFile(file, "utf-8");
				// System.out.println("Training on " + mCats[i] + "/"+
				// trainingFiles[j]);

				StringTokenizer st = new StringTokenizer(text, "\n");
				Classification classification = new Classification(mCats[i]);
				while (st.hasMoreTokens()) {
					String feat = st.nextToken();

					Classified<CharSequence> classified = new Classified<CharSequence>(
							feat, classification);
					handler.handle(classified);
				}
			}
		}
	}

	@Override
	public void visitTest(ObjectHandler<Classified<CharSequence>> handler)
			throws IOException {

		for (int i = 0; i < mCats.length; i++) {
			File classDir = new File(mTestDir, mCats[i]);
			String[] testFiles = classDir.list();
			for (int j = 0; j < testFiles.length; j++) {
				String text = Files.readFromFile(new File(classDir,
						testFiles[j]), "utf-8");
				// System.out.print("Testing on " + mCats[i] + "/"
				// +testFiles[j]+ " ");

				StringTokenizer st = new StringTokenizer(text, "\n");
				Classification classification = new Classification(mCats[i]);
				while (st.hasMoreTokens()) {
					String feat = st.nextToken();
					Classified<CharSequence> classified = new Classified<CharSequence>(
							feat, classification);
					handler.handle(classified);
				}
			}
		}

	}

	public String[] getCatogories() throws IOException {
		if (mCats == null) {
			if (mCorpusFile.isDirectory()) {
				File trainDir = new File(mCorpusFile.getAbsolutePath(), "train");
				if (trainDir.isDirectory()) {
					return trainDir.list();
				}
			} else {
				throw new IllegalArgumentException();
			}
		}

		return mCats;
	}

	public File getCorpusFile() {
		return mCorpusFile;
	}

}
