import java.util.*;
import java.io.*;


/*************************************************************************
 *
 * ISTE-612 KPT Project - US Airlines Twitter Sentiment Analysis 
 * @author: Pranav Jain
 *
 *
 *************************************************************************/

@SuppressWarnings("unchecked")
public class FinalKPTproject {

    int numOfDocs;
    int numOfClasses;
    int[] classDocs;
    String[] classText;
    int[] classTokens;
    HashSet<String> vocab;
    HashMap<String,Double>[] condProb;
    Double[] classProb;
    Scanner sc;


    /**
     * Build a Naive Bayes classifier using a training document set
     * @param trainDataFolder the training document folder
     */

    public FinalKPTproject(String trainDataFolder) throws FileNotFoundException{

        numOfDocs=0;
        preprocess(trainDataFolder);
        classProb = new Double[classText.length];

        //   System.out.println(classText.length);
        for(int i=0;i<numOfClasses;i++){
            Iterator<Map.Entry<String, Double>> iterator = condProb[i].entrySet().iterator();
            int vocabSize = vocab.size();
            while(iterator.hasNext())
            {
                Map.Entry<String, Double> entry = iterator.next();
                String token = entry.getKey();
                Double count = entry.getValue();
                count = (count+1)/(classTokens[i]+vocabSize);
                condProb[i].put(token, count);
            }
            classProb[i] = 1.0*classDocs[i]/numOfDocs;
            System.out.println(condProb[i]);

        }
    }

    public int classifystring(String doc){


        String document = doc;


        int label = 0;
        int vocabSize = vocab.size();
        double[] score = new double[numOfClasses];
        String[] tokens = document.split(" ");
        for (int i = 0; i < numOfClasses; i++) {
            score[i] = Math.log(classProb[i]);
            for (String token : tokens) {
                if(token.isEmpty())continue;
                if (condProb[i].containsKey(token)){
                    score[i] += Math.log(condProb[i].get(token));
                    condProb[i].put(token, condProb[i].get(token));
                }else{
                    score[i] += Math.log(1.0 / (classTokens[i] + vocabSize));
                }
            }
        }
        double maxScore = score[0];
        for(int i=0;i<score.length;i++){
            //   System.out.println(i);

            if(score[i]>maxScore)
                label = i;
        }

        return label;

    }


    /**
     * Classify a test doc
     * @param doc test doc
     * @return class label
     */
    public int classify(String doc) throws FileNotFoundException{

        sc = null;
        String document = new String();
        sc = new Scanner(new File(doc));

        while(sc.hasNextLine()){
            document+=sc.nextLine();
        }
        int label = 0;
        int vocabSize = vocab.size();
        double[] score = new double[numOfClasses];
        String[] tokens = document.split("[^\\w']+");
        for (int i = 0; i < numOfClasses; i++) {
            score[i] = Math.log(classProb[i]);
            for (String token : tokens) {
                if(token.isEmpty())continue;
                if (condProb[i].containsKey(token)){
                    score[i] += Math.log(condProb[i].get(token));
                    condProb[i].put(token, condProb[i].get(token));
                }else{
                    score[i] += Math.log(1.0 / (classTokens[i] + vocabSize));
                }
            }
        }
        for (int i = 0; i < score.length; i++) {
            if (score[i] > score[label])
                label = i;
        }
        return label;
    }

    /**
     * Perform Tokenization in each of the files
     * @param file documents
     * @param correctDocsClass  number of classes
     */

    public void tokenization(File file, int correctDocsClass) throws FileNotFoundException{

        Scanner scan = new Scanner(file);
        classDocs[correctDocsClass]++;
        numOfDocs++;

        while(scan.hasNextLine()){

            String[] tokens = scan.nextLine().split("[^\\w']+");

            for(String token:tokens){
                if(token.isEmpty())continue;
                if(condProb[correctDocsClass].containsKey(token)){
                    Double value = condProb[correctDocsClass].get(token);
                    value++;
                    condProb[correctDocsClass].put(token, value);
                    classTokens[correctDocsClass]++;
                }else{
                    Double value = 1.0;
                    condProb[correctDocsClass].put(token, value);
                    classTokens[correctDocsClass]++;
                    if(!vocab.contains(token))vocab.add(token);
                }
            }
        }
    }

    /**
     * Load the training documents
     * @param trainDataFolder
     */
    public void preprocess(String trainDataFolder) throws FileNotFoundException{

        File folder = new File(trainDataFolder);
        int correctDocsFolders=0;
        File[] files=folder.listFiles();
        correctDocsFolders = files.length;
        classDocs = new int[correctDocsFolders];
        classText = new String[correctDocsFolders];
        classTokens = new int[correctDocsFolders];
        condProb = new HashMap[correctDocsFolders];
        vocab = new HashSet<String>();
        numOfClasses=-1;
        for(File file:files){
            numOfClasses++;
            classText[numOfClasses]=file.getName();
            condProb[numOfClasses] = new HashMap<String,Double>();
            for(File f:file.listFiles()){
                tokenization(f,numOfClasses);
            }
        }
        numOfClasses++;
    }

    /**
     *  Classify a set of testing documents and report the accuracy
     * @param testDataFolder fold that contains the testing documents
     * @return classification accuracy
     */
    public double classifyAll(String testDataFolder) throws FileNotFoundException	{

        File testFolder = new File(testDataFolder);
        double accuracy = 0;
        int correctlyclassified=0;
        int wronglyclassified=0;
        int aClass = 0;

        for(File testClassFolder:testFolder.listFiles()){
            for(File file:testClassFolder.listFiles()){
                int pClass = classify(file.getAbsolutePath());
                if(aClass==pClass){
                    correctlyclassified++;
                }else{
                    wronglyclassified++;
                }
            }
            aClass++;
        }
        double total = correctlyclassified+wronglyclassified;
        double correctDocs = correctlyclassified;
        System.out.println("Correctly Classified: "+correctlyclassified+" out of "+(int)total);


        accuracy = (correctDocs/total)*100.00;

        return accuracy;
    }

    /**
     * Classify the set of postive tweets and rank the best airline according to the highest number of postive tweets.
     * @param pathofpostivetweets path of postive tweets folder
     */

    public void rankingAlgorithm(String pathofpostivetweets) throws FileNotFoundException{

        File[] arr = new File(pathofpostivetweets).listFiles();
        ArrayList<String> Docs = new ArrayList<String>();

        for (File filename : arr) {
            if (filename.getName().endsWith(".txt")) {
                Scanner in = new Scanner(new FileReader(filename));
                while (in.hasNext()) {
                    Docs.add(in.next());
                }

            }

        }

        String[] docs = Docs.toArray(new String[Docs.size()]);


        ArrayList<String> store = new ArrayList<String>();

        String word = "@VirginAmerica";
        String word1 = "@united";
        String word2 = "@SouthwestAir";
        String word3 = "@JetBlue";
        String word4 = "@AmericanAir";
        String word5 = "@USAirways";

        String[] words = new String[6];
        words[0] = "@VirginAmerica";
        words[1] = "@united";
        words[2] = "@SouthwestAir";
        words[3] = "@JetBlue";
        words[4] = "@AmericanAir";
        words[5] = "@USAirways";

        String[] save = null;

        store.add(word);
        store.add(word1);
        store.add(word2);
        store.add(word3);
        store.add(word4);
        store.add(word5);

        int counter = 0;
        int counter1 = 0;
        int counter2 = 0;
        int counter3 = 0;
        int counter4 = 0;
        int counter5 = 0;

        int[] count = new int[6];

        for (int i = 0; i < docs.length; i++) {
            save = docs[i].split(" ");

            for (int j = 0; j < save.length; j++) {

                if (word.equals(save[j])) {
                    counter++;
                    count[0]++;
                } else if (word1.equals(save[j])) {
                    counter1++;
                    count[1]++;
                } else if (word2.equals(save[j])) {
                    counter2++;
                    count[2]++;
                } else if (word3.equals(save[j])) {
                    counter3++;
                    count[3]++;
                } else if (word4.equals(save[j])) {
                    counter4++;
                    count[4]++;
                } else if (word5.equals(save[j])) {
                    counter5++;
                    count[5]++;
                }
            }

        }

        int max = 0;
        int pos = 0;
        for (int s = 0; s < 6; s++) {
            if (max < count[s]) {
                max = count[s];
                pos = s;
            }
        }

        System.out.println("\n"+"Best Airline");
        System.out.println("Max: " + max + " Positive Tweets: " + words[pos]+"\n");

        for (int i = 0; i < words.length - 1; i++) {
            for (int j = 0; j < words.length - 1; j++) {
                if (count[j] < count[j + 1]) {
                    int temp = count[j];
                    count[j] = count[j + 1];
                    count[j + 1] = temp;

                    String temp1 = words[j];
                    words[j] = words[j + 1];
                    words[j + 1] = temp1;
                }
            }
        }

        for (int i = 0; i < words.length; i++) {
            System.out.println(+i + " " + words[i] + " Occurence: " + count[i]);
        }


    }

    public static void main(String[] args) throws FileNotFoundException
    {
        FinalKPTproject nb = new FinalKPTproject("C:\\Users\\Pranav\\Desktop\\Testing KPT\\dataset\\train");
        System.out.println("\n<---------------Classification Result----------------->");
        System.out.println("Accuracy: "+nb.classifyAll("C:\\Users\\Pranav\\Desktop\\Testing KPT\\dataset\\test")+" %");

        nb.rankingAlgorithm("C:\\Users\\Pranav\\Desktop\\KPT project\\dataset_postive_tweets -ranking");

        Scanner scan = new Scanner(System.in);

        System.out.println("Enter a sentiment :");

        String testDoc = scan.nextLine();

        // System.out.println(nb.classifystring());



        if(nb.classifystring(testDoc) == 0){
            System.out.println("Class Prediction: Negative ");
        } else if(nb.classifystring(testDoc) == 1){
            System.out.println("Class Prediction: Neutral ");
        } else if(nb.classifystring(testDoc) == 2){
            System.out.println("Class Prediction: Positive ");
        }




    }
}


 
 
 
 
 
 
 
 
 
 
 
 
 