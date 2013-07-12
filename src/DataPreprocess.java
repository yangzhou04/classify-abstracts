import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;

public class DataPreprocess {

	public static List<String> readFeature(File f) throws IOException {
		List<String> featList = new ArrayList<String>();
		BufferedReader br = new BufferedReader(new FileReader(f));
		while (br.ready()) {
			featList.add(br.readLine());
		}

		br.close();

		return featList;
	}

	public static void writeFeature(Map<String, List<String>> labelFeatMap,
			String dir) throws IOException {

		File dst = new File(dir);
		if (dst.exists()) {
			dst.delete();
		}

		for (Iterator<String> iter = labelFeatMap.keySet().iterator(); iter
				.hasNext();) {
			String label = iter.next();

			String eol = System.getProperty("line.separator");
			File labelDir = new File(dir, label);
			if (!labelDir.exists()) {
				labelDir.mkdirs();
			}

			File feat = new File(labelDir, String.valueOf(labelFeatMap.get(
					label).size()));

			feat.createNewFile();
			BufferedWriter bw = new BufferedWriter(new FileWriter(feat));
			for (String s : labelFeatMap.get(label)) {
				bw.append(s);
				bw.append(eol);
			}
			bw.flush();
			bw.close();
		}
	}

	public static Map<String, List<String>> readTrainOrTest(File trainOrTest)
			throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(trainOrTest));
		Map<String, List<String>> labelFeatMap = new HashMap<String, List<String>>();
		while (br.ready()) {
			String line = br.readLine();
			int deliPos = line.indexOf('\t');
			String label = line.substring(0, deliPos);
			String feat = line.substring(deliPos + 1);

			if (!labelFeatMap.containsKey(label)) {
				List<String> li = new ArrayList<String>();
				li.add(feat);
				labelFeatMap.put(label, li);
			} else {
				labelFeatMap.get(label).add(feat);
			}
		}
		br.close();

		return labelFeatMap;
	}

	/**
	 * Extract features based on part of the feature rank list
	 * 
	 * @param originMap
	 *            label to feature map
	 * @param featRankList
	 *            feature rank list
	 * @param percent
	 *            using top percent of feature list
	 * @return
	 */
	public static Map<String, List<String>> extractFeature(
			Map<String, List<String>> originMap, List<String> featRankList,
			double percent) {
		if (percent >= 0 && percent <= 1) {
			int endPos = (int) (featRankList.size() * percent);
			return extractFeature(originMap, featRankList.subList(0, endPos));
		} else
			return extractFeature(originMap, featRankList);
	}

	public static Map<String, List<String>> extractFeature(
			Map<String, List<String>> originMap, List<String> featRankList) {

		Map<String, List<String>> extractedMap = new HashMap<String, List<String>>();

		for (Iterator<String> iter = originMap.keySet().iterator(); iter
				.hasNext();) {
			String label = iter.next();
			if (!extractedMap.containsKey(label))
				extractedMap.put(label, new ArrayList<String>());

			List<String> featList = originMap.get(label);
			StringBuffer sb = new StringBuffer();
			for (String feat : featList) { // feats in one line "x_1 x_2 x_3 ... x_n"
				StringTokenizer st = new StringTokenizer(feat, " ");
				while (st.hasMoreTokens()) {
					String word = st.nextToken();
					if (featRankList.contains(word)) { // one feat(word) x_i
						sb.append(word);
						sb.append(" ");
					}
				}
				extractedMap.get(label).add(sb.toString().trim());
			}
		}

		return extractedMap;
	}

	public static void main(String[] args) throws IOException,
			IllegalAccessException {
		final String DATA_FOLDER = "C:\\cygwin\\home\\Mark\\tokenization.new\\";

		System.out.println("Start processing...");
		File dataDir = new File(DATA_FOLDER); // root data directory
		if (!dataDir.exists()) {
			throw new FileNotFoundException();
		}

		File[] partDirs = dataDir.listFiles();
		List<String> featIG = new ArrayList<String>();
		List<String> featMI = new ArrayList<String>();
		List<String> featTI = new ArrayList<String>();

		for (int i = 0; i < partDirs.length; i++) {
			File[] filesInPartDirs = partDirs[i].listFiles();
			File miFile = null, igFile = null, tiFile = null;
			File trainFile = null, testFile = null;

			// route files
			for (int j = 0; j < filesInPartDirs.length; j++) {
				File f = filesInPartDirs[j];
				if (f.isFile() && f.getName().contains("mi"))
					miFile = f;
				else if (f.isFile() && f.getName().contains("ig"))
					igFile = f;
				else if (f.isFile() && f.getName().contains("tfidf"))
					tiFile = f;
				else if (f.isFile() && f.getName().contains("part1.csv"))
					trainFile = f;
				else if (f.isFile() && f.getName().contains("part2.csv"))
					testFile = f;
				else
					throw new IllegalAccessException();
			}

			// read feature
			featIG = readFeature(igFile);
			featMI = readFeature(miFile);
			featTI = readFeature(tiFile);

			// read train and test
			Map<String, List<String>> trainMap = readTrainOrTest(trainFile);
			Map<String, List<String>> testMap = readTrainOrTest(testFile);

			// extract feature based on ig, mi and tfidf
			for (int j = 1; j < 10; j++) {
				double percent = 0.1 * j;

				Map<String, List<String>> exTrainByIGMap = extractFeature(
						trainMap, featIG, percent);
				Map<String, List<String>> exTrainByMIMap = extractFeature(
						trainMap, featMI, percent);
				Map<String, List<String>> exTrainByTIMap = extractFeature(
						trainMap, featTI, percent);

				Map<String, List<String>> exTestByIGMap = extractFeature(
						testMap, featIG);
				Map<String, List<String>> exTestByMIMap = extractFeature(
						testMap, featMI);
				Map<String, List<String>> exTestByTIMap = extractFeature(
						testMap, featTI);

				// write extracted feature to file
				writeFeature(exTrainByIGMap, partDirs[i].getAbsolutePath()
						+ "\\" + percent + "\\ig\\train");
				writeFeature(exTestByIGMap, partDirs[i].getAbsolutePath()
						+ "\\" + percent + "\\ig\\test");

				writeFeature(exTrainByMIMap, partDirs[i].getAbsolutePath()
						+ "\\" + percent + "\\mi\\train");
				writeFeature(exTestByMIMap, partDirs[i].getAbsolutePath()
						+ "\\" + percent + "\\mi\\test");

				writeFeature(exTrainByTIMap, partDirs[i].getAbsolutePath()
						+ "\\" + percent + "\\ti\\train");
				writeFeature(exTestByTIMap, partDirs[i].getAbsolutePath()
						+ "\\" + percent + "\\ti\\test");
			}
			System.out.println(partDirs[i] + "....Done!");
		}
		System.out.println("Completed!");
	}
}
