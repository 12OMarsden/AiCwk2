from utils import *
import numpy as np
import csv
import scipy.stats as stats


class DataSet:
    """

    d.examples   A list of examples. Each one is a list of attribute values.
    d.attr_names List of mnemonic names for corresponding attrs.
    d.target     The attribute that a learning algorithm will try to predict.
                 By default the final attribute.
    d.name       Name of the data set.
    """

    def __init__(self, examples=None, attrs=None, attr_names=None, name = '', target=-1):

        self.name = name

        # initialize .examples from string or list or data directory

        if examples is None:
            # opening the CSV file
            with open((name + '.csv'),'r') as file:
                # reading the CSV file
                csvFile = csv.reader(file)
                attr_names = next(csvFile)
                self.examples = list(csv.reader(file))
        else:
            self.examples = examples


        # attrs are the indices of examples, unless otherwise stated.
        if self.examples is not None and attrs is None:
            attrs = list(range(len(self.examples[0])))


        self.attrs = attrs

        # initialize .attr_names from string, list, or by default
        if isinstance(attr_names, str):
            self.attr_names = attr_names.split()
        else:
            self.attr_names = attr_names or attrs



        self.target = self.attr_num(target)

        self.inputs = [a for a in self.attrs if a != self.target]
        # find possible range of values for attributes
        self.values = list(map(unique, zip(*self.examples)))


    def attr_num(self, attr):
        """Returns the number used for attr, which can be a name, or -n .. n-1."""
        if isinstance(attr, str):
            return self.attr_names.index(attr)
        elif attr < 0:
            return len(self.attrs) + attr
        else:
            return attr


    def sanitize(self, example):
        """Return a copy of example, with non-input attributes replaced by None."""
        return [attr_i if i in self.inputs else None for i, attr_i in enumerate(example)]

    def classes_to_numbers(self, classes=None):
        """Converts class names to numbers."""
        if not classes:
            # if classes were not given, extract them from values
            classes = sorted(self.values[self.target])
        for item in self.examples:
            item[self.target] = classes.index(item[self.target])

    def remove_examples(self, value=''):
        """Remove examples that contain given value."""
        self.examples = [x for x in self.examples if value not in x]
        self.update_values()

    def split_values_by_classes(self):
        """Split values into buckets according to their class."""
        buckets = defaultdict(lambda: [])
        target_names = self.values[self.target]

        for v in self.examples:
            item = [a for a in v if a not in target_names]  # remove target from item
            buckets[v[self.target]].append(item)  # add item to bucket of its class

        return buckets

    def __repr__(self):
        return '<DataSet({}): {:d} examples, {:d} attributes>'.format(self.name, len(self.examples), len(self.attrs))



def err_ratio(predict, dataset, examples=None):
    """
    Return the proportion of the examples that are NOT correctly predicted.
    verbose - 0: No output; 1: Output wrong; 2 (or greater): Output correct
    """
    examples = examples or dataset.examples
    if len(examples) == 0:
        return 0.0
    right = 0
    for example in examples:
        desired = example[dataset.target]
        output = predict(dataset.sanitize(example))
        if output == desired:
            right += 1

    return 1 - (right / len(examples))


def grade_learner(predict, tests):
    """
    Grades the given learner based on how many tests it passes.
    tests is a list with each element in the form: (values, output).
    """
    return mean(int(predict(X) == y) for X, y in tests)


def train_test_split(dataset, start=None, end=None, test_split=None):
        """
        If you are giving 'start' and 'end' as parameters,
        then it will return the testing set from index 'start' to 'end'
        and the rest for training.
        If you give 'test_split' as a parameter then it will return
        test_split * 100% as the testing set and the rest as
        training set.
        """
        examples = dataset.examples
        if test_split is None:
            train = examples[:start] + examples[end:]
            val = examples[start:end]
        else:
            total_size = len(examples)
            val_size = int(total_size * test_split)
            train_size = total_size - val_size
            train = examples[:train_size]
            val = examples[train_size:total_size]

        train_set = DataSet(examples = train, attr_names = dataset.attr_names,attrs = dataset.attrs,target = dataset.target)
        val_set = DataSet(examples = val, attr_names = dataset.attr_names,attrs = dataset.attrs,target = dataset.target)

        return train_set, val_set





def PluralityLearner(dataset):
    """
    A very dumb algorithm: always pick the result that was most popular
    in the training data. Makes a baseline for comparison.
    """
    most_popular = mode([e[dataset.target] for e in dataset.examples])

    def predict(example):
        """Always return same result: the most popular from the training set."""
        return most_popular

    return predict

class DecisionFork:
    """
    A fork of a decision tree holds an attribute to test, and a dict
    of branches, one for each of the attribute's values.
    """

    def __init__(self, attr, attr_name=None, default_child=None, branches=None,parent=None):
        """Initialize by saying what attribute this node tests."""
        self.attr = attr
        self.attr_name = attr_name or attr
        self.default_child = default_child
        self.branches = branches or {}
        self.pos=0
        self.neg=0
        self.parent_node = parent

    def __call__(self, example,target=None):
        """Given an example, classify it using the attribute and the branches."""
        attr_val = example[self.attr]

        if(target):
            if(example[target] == "Yes"):
                self.pos = self.pos+1
            else:
                self.neg = self.neg+1

        if attr_val in self.branches:
            return self.branches[attr_val](example,target)
        else:
            print("attr not found ",attr_val)
            # return default class when attribute is unknown
            return self.default_child(example)

    def clear_count(self):
        self.pos = 0
        self.neg = 0
        return(0)

    def add(self, val, subtree):
        """Add a branch. If self.attr = val, go to the given subtree."""
        self.branches[val] = subtree

    def display(self, indent=0):
        name = self.attr_name
        print('Test', name)
        for (val, subtree) in self.branches.items():
            print(' ' * 4 * indent, name, '=', val, '==>', end=' ')
            subtree.display(indent + 1)

    def __repr__(self):
        return 'DecisionFork({0!r}, {1!r}, {2!r})'.format(self.attr, self.attr_name, self.branches)

class DecisionLeaf:
    """A leaf of a decision tree holds just a result."""

    def __init__(self, result,parent=None):
        self.pos=0
        self.neg=0
        self.result = result
        self.parent_node = parent

#    def __call__(self, example):
#        return self.result

    def __call__(self, example,target=None):

        if(target):
            if(example[target] == "Yes"):
                self.pos= self.pos+1
            else:
                self.neg= self.neg+1

        return self.result

    def clear_count(self):
        self.pos = 0
        self.neg = 0
        return(0)


    def display(self,indent=0):
        print('RESULT =', self.result)

    def __repr__(self):
        return repr(self.result)

def DecisionTreeLearner(dataset):

    target, values = dataset.target, dataset.values

    def decision_tree_learning(examples, attrs, parent_examples=()):
        if len(examples) == 0:
            return plurality_value(parent_examples)
        if all_same_class(examples):
            return DecisionLeaf(examples[0][target])
        if len(attrs) == 0:
            return plurality_value(examples)
        A = choose_attribute(attrs, examples)
        tree = DecisionFork(A, dataset.attr_names[A], plurality_value(examples))
        for (v_k, exs) in split_by(A, examples):
            subtree = decision_tree_learning(exs, remove_all(A, attrs), examples)
            tree.add(v_k, subtree)
        return tree

    def plurality_value(examples):
        """
        Return the most popular target value for this set of examples.
        (If target is binary, this is the majority; otherwise plurality).
        """
        np.random.seed(1915)
        popular = argmax_random_tie(values[target], key=lambda v: count(target, v, examples))

        return DecisionLeaf(popular)

    def count(attr, val, examples):
        """Count the number of examples that have example[attr] = val."""
        return sum(e[attr] == val for e in examples)

    def all_same_class(examples):
        """Are all these examples in the same target class?"""
        class0 = examples[0][target]
        return all(e[target] == class0 for e in examples)

    def choose_attribute(attrs, examples):
        """Choose the attribute with the highest information gain."""
        np.random.seed(1915)
        return argmax_random_tie(attrs, key=lambda a: information_gain(a, examples))

    def information_gain(attr, examples):
        """Return the expected reduction in entropy from splitting by attr."""

        def I(examples):
            return information_content([count(target, v, examples) for v in values[target]])

        n = len(examples)
        remainder = sum((len(examples_i) / n) * I(examples_i) for (v, examples_i) in split_by(attr, examples))
        return I(examples) - remainder

    def split_by(attr, examples):
        """Return a list of (val, examples) pairs for each val of attr."""
        return [(v, [e for e in examples if e[attr] == v]) for v in values[attr]]

    return decision_tree_learning(dataset.examples, dataset.inputs)


def information_content(values):
    """Number of bits to represent the probability distribution in values."""
    probabilities = normalize(remove_all(0, values))
    return sum(-p * np.log2(p) for p in probabilities)

#Task 4c
def deviation(value,parent_pos,parent_neg):
    ##actual counts of exmaples at this node are given by value.pos and value.neg
    ##actula counts at its parents are parent.pos and parent.neg
    ##function must return the sqaured difference between the actual and expected counts.
    deviation = 0

    if(value.pos == 0 and value.neg == 0):
        return 0

    #Insert code here


    return (deviation)

def replaceFork(parent,leaf):
    if(parent.parent_node == None): # tries to handle removal of top node (has no parents)
        for key, value in list(parent.branches.items()): #make all branches same leaf
            parent.branches[key] = leaf
    else:
        for key, value in list(parent.parent_node.branches.items()):
            if value == parent:
                parent.parent_node.branches[key] = leaf
                return 1
    return 0

def order(tree):

    def decisiontree_iterator(parent):
        ''' This function accepts a node as an argument
        and iterate over all values of its children to clear examples counts
        '''

        parent.branches = dict(sorted(parent.branches.items(), key = lambda kv: kv[0]))


        # Iterate over all key-value pairs of DecisionFrok argument
        for key, value in list(parent.branches.items()):
                if isinstance(value,DecisionFork):
                    yield from decisiontree_iterator(value)

    all_nodes = list(decisiontree_iterator(tree))


    return tree


def clear_counts(tree):

    def decisiontree_iterator(parent):
        ''' This function accepts a node as an argument
        and iterate over all values of its children to clear examples counts
        '''
        # Iterate over all key-value pairs of DecisionFrok argument
        for key, value in list(parent.branches.items()):
                value.clear_count()
                if isinstance(value,DecisionFork):
                    yield from decisiontree_iterator(value)

    all_nodes = list(decisiontree_iterator(tree))

#Task 4d
def evaluate(predict,dataset, examples = None):
    """
    Return the proportion of the examples that are NOT correctly predicted.

    """
    def decisiontree_iterator(parent):
        ''' This function accepts a node as an argument
        and iterate over all values of its children
        '''
        p_value = 0
        DELTA = -1.0
        further_fork = False #used to flag if fork only has leaf nodes

        # Iterate over all key-value pairs of DecisionFrok argument
        for key, value in list(parent.branches.items()):
            value.parent_node = parent #makes it easier to prune back
            if isinstance(value,DecisionFork):
                further_fork = True
                #yield from decisiontree_iterator(value)
                p_value,DELTA = decisiontree_iterator(value)


        # Is a fork with only leaf nodes therefore could be pruned
        if further_fork == False:

            if(parent.pos > 0 and parent.neg > 0):

                DELTA = sum(deviation(value,parent.pos,parent.neg) for key, value in parent.branches.items())

                #Insert code here
                #calculate p_value using the stats.chi2.cdf function.
                #The degree of freedom (num of variable) is the number of branches at the parent.



                print("chisquare-score is:", DELTA, " and p value is:", p_value)

                if p_value <= 0.05:
                    print("Null Hypothesis is rejected.")
                else:
                    print("Failed to reject the Null hypothesis.")
                    print("Pruning")
                    #can prune parent
                    if(parent.pos> parent.neg):
                        replaceFork(parent,DecisionLeaf("Yes"))
                    else:
                        replaceFork(parent,DecisionLeaf("No"))

        return(p_value,DELTA)



    examples = examples or dataset.examples
    if len(examples) == 0:
        return 0.0
    right = 0

    target = dataset.target

    #predict outcome for each exmaple
    for example in examples:
        desired = example[dataset.target]
        output = predict(example,target)
        if output == desired:
            right += 1


    p_value,DELTA = decisiontree_iterator(predict)


    return (p_value,DELTA,1 - (right / len(examples)))



def RestaurantDataSet(examples=None):
   """
   [Figure 19.3]
   Build a DataSet of Restaurant waiting examples.
   """
   return DataSet(name='restaurant1', target='Wait', examples=examples,
                        attr_names='Alternate Bar Fri/Sat Hungry Patrons Raining Reservation WaitEstimate Wait')

restaurant = RestaurantDataSet()


def T(attr_name, branches):
    branches = {value: (child if isinstance(child, DecisionFork) else DecisionLeaf(child))
                for value, child in branches.items()}
    return DecisionFork(attr = restaurant.attr_num(attr_name),attr_name=attr_name,default_child = print, branches = branches)


waiting_decision_tree = T('Patrons',
                          {'None': 'No', 'Some': 'Yes',
                           'Full': T('WaitEstimate',
                                     {'>60': 'No', '0-10': 'Yes',
                                      '30-60': T('Alternate',
                                                 {'No': T('Reservation',
                                                          {'Yes': 'Yes',
                                                           'No': T('Bar', {'No': 'No',
                                                                           'Yes': 'Yes'})}),
                                                  'Yes': T('Fri/Sat', {'No': 'No', 'Yes': 'Yes'})}),
                                      '10-30': T('Hungry',
                                                 {'No': 'Yes',
                                                  'Yes': T('Alternate',
                                                           {'No': 'Yes',
                                                            'Yes': T('Raining',
                                                                     {'No': 'No',
                                                                      'Yes': 'Yes'})})})})})



def SyntheticRestaurantPruneTest(n=100):
  """Generate a DataSet with n examples."""
  np.random.seed(4000)

  def gen():
    example = list(map(np.random.choice, restaurant.values))
    example[restaurant.target] = waiting_decision_tree(example)


    rand = np.random.random_sample()
    if rand >= 0.2:
        rand = np.random.random_sample()
        if rand >= 0.5:
                if example[5] == "Yes":
                    example[5] = "No"
                else:
                    if example[1] == "Yes":
                        example[1] = "No"

    return example

  return RestaurantDataSet([gen() for _ in range(n)])

def SyntheticRestaurantTest(n=100):
  """Generate a DataSet with n examples."""
  np.random.seed(4000)

  def gen():
    example = list(map(np.random.choice, restaurant.values))

    rand = np.random.random_sample()
    if rand >= 1.1:
        if example[restaurant.target] == "Yes":
            example[restaurant.target] = "No"

        else:
            example[restaurant.target] == "Yes"

    return example

  return RestaurantDataSet([gen() for _ in range(n)])


def SyntheticRestaurant(n=100):
  """Generate a DataSet with n examples."""

  np.random.seed(2000)


  def gen():

    example = list(map(np.random.choice, restaurant.values))

    example[restaurant.target] = waiting_decision_tree(example)
    return example

  return RestaurantDataSet([gen() for _ in range(n)])

#TASK 1

def learn_tennis_tree(filename):
    ## function should create a decision tree from the file named filename
    ## returns the data set used to create the tree and the learnt decision tree

    dataSet = None
    tree = None


    #insert code here


    return(dataSet,tree)

#TASK 2

def test_tennis_tree(filename):
    ## function should split the data provided by filename into a traing and test set.
    ## learn a decision tree from the training set
    ## test the tree on the test and evaluate its performance
    ## returns a the training and test sets, the decion tree and the error rate achieved.

    trainSet = None
    testSet = None
    tree = None
    error = 0

    #insert code here

    return(trainSet,testSet,tree,error)

# TASK 3a
def genSyntheticTrainSet():
    ##function generates a synthetic data set using the SytheticRestatuant method
    ##returns the dataset created
    data = None

    #insert code here


    return(data)

# TASK 3b
def genSyntheticTestSet():
    ##function generates a synthetic data set using the SytheticRestatuantTest method
    ##returns the dataset created
    data = None

    #insert code here


    return (data)

#TASK 3c
def train_restaurant_tree(trainSet, testSet, N=200):
    ## function should learn decision trees using different quantities of the training set (trainSet) from 1 to N
    ## where N should be the total size of the training set (trainSet)
    ## and test each tree on the whole test set (testSet) provided
    ## the function should return the final tree obtained using all 200 samples,
    ## and the minimum size of the training set (samples) required to achieve the same error rate as achieved using all 200 training samples.

    tree = None
    samples_required = 0


    #insert code here


    return(tree,samples_required)

#TASK 3d
def train_tree(trainSet, testSet):
    ## function should learn a decision tree the training set (trainSet)
    ## and test the tree on the whole test set (testSet)
    ## the method should return the tree, and the error rate achieved.
    tree = None
    error = 0

    #insert code here


    return(tree,error)

#TASK 4a
def genPruneTestSet():
    ##function generates a synthetic data set using the SytheticRestaruantPruneTest method
    ##returns the dataset created
    data = None

    #insert code here


    return(data)

#TASK 4b
def prune_tree(tree,testSet):
    ##function should prune the decison tree (tree) using the evaluate method as many times as required when evaluated using testSet.
    ##the function must return the testSet used, the p_value, K and error rates of the final tree (tree) returned from the evalaute function.

    p_value = 0
    error_rate = 0
    delta = 1.0

    #insert code here


    return(testSet,p_value,delta,tree,error_rate)



if __name__ == "__main__":
    filename = "./tennis"
    tennis_dataSet,tree = learn_tennis_tree(filename) #task 1
    error_rate = test_tennis_tree(filename) #task 2
    print("Error_rate ",error_rate)
    train_set = genSyntheticTrainSet() #task 3a
    test_set = genSyntheticTestSet() #task 3b
    restaurant_tree,errors = train_restaurant_tree(train_set,test_set) #task 3c

    tree,error_rate = train_tree(train_set,test_set) #task 2d

    testData = genPruneTestSet() #task 4a
    testData,p_value,delta,pruned_tree,error = prune_tree(tree,testData) #task 4b,c and d
    print("pruned error rate ",error)
