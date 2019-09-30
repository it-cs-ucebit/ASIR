 
 import weka.classifiers.trees.J48;
 import weka.core.converters.ConverterUtils.DataSource;
 import weka.classifiers.Evaluation;
 import java.util.Random;
 import weka.core.Instances;

 

public class WekaTest {
    public static void main(String[] args) throws Exception {
        
  
    
 DataSource source = new DataSource("C:\\\\Program Files\\\\WEKA_HOME\\\\data\\\\diabetes.arff");
 Instances data = source.getDataSet();
 // setting class attribute if the data format does not provide this information
 // For example, the XRFF format saves the class attribute information as well
 if (data.classIndex() == -1)
   data.setClassIndex(data.numAttributes() - 1);
    
     
 String[] options = new String[1];
 options[0] = "-U";            // unpruned tree
 J48 tree = new J48();         // new instance of tree
 tree.setOptions(options);     // set the options
 tree.buildClassifier(data);   // build classifier 
    
    
 Evaluation eval = new Evaluation(data);
 eval.crossValidateModel(tree, data, 10, new Random(1));
    System.out.println(eval.toSummaryString("\nResults\n======\n", false));
    }
}