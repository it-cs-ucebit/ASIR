import weka.classifiers.trees.J48;
 import weka.core.converters.ConverterUtils.DataSource;
 import weka.classifiers.Evaluation;
 import java.util.Random;
 import weka.core.Instances;
import weka.attributeSelection.AttributeSelection;
 import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.classifiers.meta.AttributeSelectedClassifier;

public class WekaTest1 {
    public static void main(String[] args) throws Exception {
        
  
    
 DataSource source = new DataSource("C:\\\\Program Files\\\\WEKA_HOME\\\\data\\\\diabetes.arff");
 Instances data = source.getDataSet();
 // setting class attribute if the data format does not provide this information
 // For example, the XRFF format saves the class attribute information as well
 if (data.classIndex() == -1)
   data.setClassIndex(data.numAttributes() - 1);
    
       
  AttributeSelectedClassifier classifier = new AttributeSelectedClassifier();
  CfsSubsetEval eval = new CfsSubsetEval();
  GreedyStepwise search = new GreedyStepwise();
  search.setSearchBackwards(true);
  J48 base = new J48();
  classifier.setClassifier(base);
  classifier.setEvaluator(eval);
  classifier.setSearch(search);
  // 10-fold cross-validation
  Evaluation evaluation = new Evaluation(data);
  evaluation.crossValidateModel(classifier, data, 10, new Random(1));
  System.out.println(evaluation.toSummaryString());
 
    }
}