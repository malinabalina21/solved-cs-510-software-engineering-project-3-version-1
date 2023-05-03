Download Link: https://assignmentchef.com/product/solved-cs-510-software-engineering-project-3-version-1
<br>



The whole project 3 is meant to be 22% of your final grade We plan to have the following breakdown for project 3: part (I): 30%, part (II): 40%, part (III): 30%, and part (IV): Bonus

You will work in a group of 1-2 members for this project. We expect that you will be in the same group as Project 2. But if you need to make changes, please notify Yi Sun by the <strong>Group Sign-Up Due Date </strong>above.

Please download proj-skeleton.tar.gz from the course repo, which you will need for this project. The skeleton contains the necessary source code and test cases for the project.

<strong>Training deep learning models may take hours or longer. Please start early; otherwise, there may not be enough machine time to finish the experiments. Also, the servers get busier/slower when many groups use them at the end when the project is due</strong>.

You are expected to use mc18.cs.purdue.edu or cuda.cs.purdue.edu machines to work on your project. Your home directory may not have enough space for the project. <strong>Use /scratch instead, which has enough space for the project</strong>. Remember to remove your data if you no longer need it. Several of the resources required for this project are already installed on these servers, but can also be downloaded independently. I expect each group to work on the project independently (discussion and collaboration among group)

<strong>Submission Instructions:</strong>

Go to Blackboard → Project 3 to submit your answer. Submit only one file in .tar.gz format. Please name your file

<em>&lt;</em>FirstName<em>&gt;</em>–<em>&lt;</em>LastName<em>&gt;</em>–<em>&lt;</em>Username<em>&gt;</em>.tar.gz

For example, use John-Smith-jsmith.tar.gz if you are John Smith with username jsmith. The .tar.gz file should contain the following items:

<ul>

 <li>a single pdf file “proj3 sub.pdf”. The first page must include your full name, and your Purdue email address. Your PDF file should contains your results for question I and a report of the improvements you tried for questions II, III, and IV as well as your results.</li>

 <li>a directory “q2” that contains your code (source code only; no binaries, datasets or trained models) for Question</li>

</ul>

II

<ul>

 <li>a directory “q3” that contains your code (source code only; no binaries, datasets or trained models) for Question</li>

</ul>

III

<ul>

 <li>(optional) a directory “q4” that contains your code (source code only; no binaries, datasets or trained models) for the competition for Question IV</li>

</ul>

If you use new libraries for questions II, III, and IV, also include a requirements.txt that contains the list of libraries used. You can use <em>pipfreeze &gt; requirements.txt </em>and include it with your source code.

You can submit multiple times. After submission, <strong>please view your submissions to make sure you have uploaded the right files/versions</strong>.

<strong>Building Line-level Defect Detection Models</strong>

In this project, you are expected to learn how to build a defect prediction model for software source code from scratch. You are required to apply deep-learning techniques, e.g., classification, tokenization, embedding, etc., to build more accurate prediction models with the dataset provided.

<h1>Background</h1>

Line-level Defect classifiers predict which lines in a file are likely to be buggy.

A typical line-level defect prediction using deep-learning consists of the following steps:

<ul>

 <li>Data extraction and labeling: Mining buggy and clean lines from a large dataset of software changes (usually GitHub).</li>

 <li>Tokenization and pre-processing: Deep learning algorithms take a vector as input. Since source code is text, it needs to be tokenized and transformed into a vector before being fed to the model.</li>

 <li>Model Building: Using the tokenized data and labels to train a deep learning classifier. Many different classifiers have been shown to work for text input (RNNs and CNNs). Most of these models can be built using TensorFlow.</li>

 <li>Defect Detection: Unlabelled instances (i.e., line of codes or files) are fed to the trained model that will classify them as buggy or clean.</li>

</ul>

<h2>Evaluation Metrics</h2>

Metrics, i.e., <em>Precision</em>, <em>Recall</em>, and <em>F</em>1, are widely used to measure the performance of defect prediction models. Here is a brief introduction:

(1)

(2)

(3)

These metrics rely on four main numbers: <em>true positive</em>, <em>false positive</em>, <em>true negative</em>, and <em>false negative</em>. True positive is the number of predicted defective instances that are truly defective, while false positive is the number of predicted defective ones that are actually not defective. True positive records the number of predicted non-defective instances that are actually defective, while false negative is the number of predicted non-defective instances that are actually defective. F1 is the weighted average of precision and recall.

These methods are threshold-dependent and are not the best to evaluate binary classifiers. In this project, we will also use the Receiver operating characteristic curve (ROC curve) and its associated metric, Area under the ROC curve (AUC) to evaluate our trained models independently from any thresholds. The ROC curve is created by plotting the true positive rate (or recall, see definition above) against the false positive rate at various threshold settings.

(4)

<h1>(I)- Using TensorFlow to build a simple classification model</h1>

Part I will guide you through building a simple bidirectional LSTM model, while part II and III will let you explore different ways to improve it.

CS Linux Servers have the environment ready to use. The following instructions assume using one of these machines unless stated otherwise.

(We’ve tested on mc18.cs.purdue.edu and cuda.cs.purdue.edu. Other mc machines may or may not work)

The environment uses Python 3 and virtualenv. For more information on how to use virtualenv, please look at the virtualenv documentation (<a href="https://virtualenv.pypa.io/en/latest/userguide/">https://virtualenv.pypa.io/en/latest/userguide/</a><a href="https://virtualenv.pypa.io/en/latest/userguide/">) </a><strong>source </strong>/homes/cs510/project −3/venv/bin/activate

*If you work on your own machine, after you created your virtualenv session and activated it, you can install the required library using the requirements.txt file we provided:

pip install −−upgrade pip pip install −r requirements . txt

<strong>0.1       Load the Input Data:</strong>

Since the dataset is quite large (9GB uncompressed), we put it in /homes/cs510/project-3/data folder on the servers.

You can also download it from <a href="https://drive.google.com/file/d/1MTBAQ-Nw2yPr8drU-cQPae17eSHvz-4j">https://drive.google.com/file/d/1MTBAQ-Nw2yPr8drU-cQPae17eSHvz-4j</a> if you want to work on your own machine.

If you prefer to work on your own machine, you will need to download the data and update the path in tokenization.py

The training, validation and test data are made available in pickled Pandas dataframes, respectively in train.pickle, valid.pickle, and test.pickle

The panda dataframes consists of 4 columns:

<ul>

 <li><strong>instance</strong>: the line under test</li>

 <li><strong>contextbefore</strong>: the context of the line under test right before the line. In this question, the context before consists of all the lines in the functions before the tested line.</li>

 <li><strong>contextafter</strong>: the context of the line under test right after the line. In this question, the context after consists of all the lines in the functions after the tested line.</li>

 <li><strong>is </strong><strong>buggy</strong>: the label of the line tested. 0 means the line is not buggy, 1 means the line is buggy.</li>

</ul>

The first step is to load the data and tokenize it. To load the data, use the following code (modify the paths if necessary):

<em># Load the data :</em>

with <strong>open</strong>( ’data/train . pickle ’ , ’rb ’ ) as handle : train = pickle . load ( handle )

with <strong>open</strong>( ’data/valid . pickle ’ , ’rb ’ ) as handle : valid = pickle . load ( handle )

with <strong>open</strong>( ’data/ test . pickle ’ ,                                ’rb ’ ) as handle :

test = pickle . load ( handle )

The custom tokenizer implemented in tokenization.py is a basic java tokenizer from the javalang library (<a href="https://github.com/c2nes/javalang">https:// </a><a href="https://github.com/c2nes/javalang">github.com/c2nes/javalang</a><a href="https://github.com/c2nes/javalang">)</a> that is enhanced to also abstract string literals and numbers different from 0 and 1.

<em># Tokenize and shape our input : </em><strong>def </strong>custom tokenize ( string ) :

<strong>try </strong>: tokens = <strong>list </strong>( javalang . tokenizer . tokenize ( string ) )

<strong>except </strong>: <strong>return </strong>[ ]

values = [ ] <strong>for </strong>token <strong>in </strong>tokens :

<em># Abstract strings </em><strong>if </strong>’” ’ <strong>in </strong>token . value <strong>or </strong>” ’” <strong>in </strong>token . value :

values . append( ’$STRING$ ’ )

<em># Abstract numbers ( except 0 and 1) </em><strong>elif </strong>token . value . isdigit () <strong>and int </strong>( token . value ) <em>&gt; </em>1:

values . append( ’$NUMBER$’ )

<em>#other wise : get the value </em><strong>else </strong>: values . append( token . value )

<strong>return </strong>values

<strong>def </strong>tokenize df ( df ) :

df [ ’ instance ’ ] = df [ ’ instance ’ ] . <strong>apply</strong>(<strong>lambda </strong>x: custom tokenize (x) ) df [ ’ context before ’ ] = df [ ’ contextbefore ’ ] . <strong>apply</strong>(<strong>lambda </strong>x: custom tokenize (x) ) df [ ’ context after ’ ] = df [ ’ contextafter ’ ] . <strong>apply</strong>(<strong>lambda </strong>x: custom tokenize (x) )

<strong>return </strong>df

test = tokenize df ( test ) train = tokenize df ( train ) valid = tokenize df ( valid )

with <strong>open</strong>( ’data/tokenizedtrain . pickle ’ , ’wb’ ) as handle : pickle .dump( train , handle , protocol=pickle .HIGHEST PROTOCOL)

with <strong>open</strong>( ’data/tokenizedvalid . pickle ’ , ’wb’ ) as handle : pickle .dump( valid , handle , protocol=pickle .HIGHEST PROTOCOL)

with <strong>open</strong>( ’data/ tokenizedtest . pickle ’ ,                                         ’wb’ ) as handle :

pickle .dump( test , handle ,                       protocol=pickle .HIGHEST PROTOCOL)

Loading the data and tokenizing it can be done by running the script:

python tokenization .py

The tokenized dataset will be saved in the data folder under proj-skeleton (not data folder under /homes/cs510/project3). You can change it if necessary. The tokenization should take about 80 minutes.

<h1>0.2      Preprocessing data</h1>

Once we have the tokenized data, we need to transform them into vectors before feeding them to the deep learning model.

This part can be done by running the script:

python preprocess .py

It will do the transformation and save the transformed data (x train.pickle, etc.) under data folder.

For this question, we represent each instance as one vector of tokens: tokenized context before, <em>&lt; START &gt;</em>, tokenized line under test, <em>&lt; END &gt;</em>, tokenized context after

The tokens <em>&lt; START &gt; </em>and <em>&lt; END &gt; </em>indicates when the line under test starts.

For this question, we will only keep 50,000 training instances to save time. You can try to use larger dataset (1 million or more) in part II-IV.

Loading tokenized data and reshaping the input:

<em># Loading tokenized data </em>with <strong>open</strong>( ’data/tokenizedtrain . pickle ’ , ’rb ’ ) as handle : train = pickle . load ( handle )

with <strong>open</strong>( ’data/tokenizedvalid . pickle ’ , ’rb ’ ) as handle : valid = pickle . load ( handle )

with <strong>open</strong>( ’data/ tokenizedtest . pickle ’ , ’rb ’ ) as handle : test = pickle . load ( handle )

<em># Reshape instances :</em>

<strong>def </strong>reshape instances ( df ) :

df [ ”input” ] = df [ ”context before” ] . <strong>apply</strong>(<strong>lambda </strong>x: ” ” . join (x) ) + ” <em>&lt;</em>START<em>&gt; </em>” + df [ ”instance” ] .

<strong>apply</strong>(<strong>lambda </strong>x: ” ” . join (x) ) + ” <em>&lt;</em>END<em>&gt; </em>” + df [ ” context after ” ] . <strong>apply</strong>(<strong>lambda </strong>x: ” ” . join (x) ) X df = [ ]

Y df = [ ]

<strong>for </strong>index , rows <strong>in </strong>df . iterrows () : X df . append(rows . <strong>input</strong>)

Y df . append(rows . is buggy ) <strong>return </strong>X df , Y df

Xtrain ,       Y train = reshape instances ( train ) Xtest ,          Y test = reshape instances ( test )

Xvalid ,                         Y valid = reshape instances ( valid )

Xtrain = X train [:50000]

Ytrain = Y train [:50000]

Xtest = X test [:25000]

Ytest = Ytest [:25000]

Xvalid = Xvalid [:25000]

Yvalid = Yvalid [:25000]

Since the deep learning model takes a fixed-length vector of numbers as input, we use the training set to build a vocabulary that maps each token to a number. Then we encode our training, testing and validation instances and created vectors of fixed length representing the encoded instances. We limit the size of an instance to 1,000 tokens. In part II-IV, you might want to experiment with different vector sizes.

<em># Build vocabulary and encoder from the training instances </em>maxlen = 1000 vocabulary set = <strong>set </strong>() <strong>for </strong>data <strong>in </strong>X train : vocabulary set . update(data . split () )

vocab size = <strong>len</strong>( vocabulary set )

<strong>print</strong>( vocab size )

<em># Encode training , valid and test instances </em>encoder = tfds . features . text . TokenTextEncoder( vocabulary set )

<strong>def </strong>encode( text ) :

encoded text = encoder . encode( text )

<strong>return </strong>encoded text

Xtrain = <strong>list </strong>(<strong>map</strong>(<strong>lambda </strong>x: encode(x) , Xtrain ) ) Xtest = <strong>list </strong>(<strong>map</strong>(<strong>lambda </strong>x: encode(x) , Xtest ) )

Xvalid = <strong>list </strong>(<strong>map</strong>(<strong>lambda </strong>x: encode(x) , Xvalid ) )

Xtrain = padsequences ( Xtrain , maxlen=maxlen) Xtest = padsequences ( Xtest , maxlen=maxlen)

Xvalid = padsequences ( Xvalid , maxlen=maxlen)

<h1>0.3      Training the model</h1>

Training and evaluation of the model is done by train and test.py

For our first model, we will try to train a two layers bidirectional RNN model using LSTM layers. RNNs have been known to work well with text data. A tutorial showing how to create a basic RNN model with TensorFlow is available on <a href="https://www.tensorflow.org/tutorials/text/text_classification_rnn">https://www.tensorflow.org/tutorials/text/text_classification_rnn</a>

Our model will be defined as followed:

<em># Model Definition </em>model = tf . keras . Sequential ([ tf . keras . layers . Embedding( encoder . vocab size , 64) , tf . keras . layers . Bidirectional ( tf . keras . layers .LSTM(64 , return sequences=True) ) , tf . keras . layers . Bidirectional ( tf . keras . layers .LSTM(32) ) , tf . keras . layers . Dense(64 , activation=’ relu ’ ) , tf . keras . layers . Dropout (0.5) ,

tf . keras . layers . Dense(1 ,                   activation=’ sigmoid ’ )

])

model . <strong>compile</strong>( loss=’ binary crossentropy ’ , optimizer=tf . keras . optimizers .Adam(1e−4) , metrics=[ ’ accuracy ’ ])

model .summary()

Since the data is pretty large, we might not be able to fit an embedding for the entire dataset in memory. Therefore, we need to build a batch generator to generate the embedding for the input data on the fly.

<em># Building generators </em><strong>class </strong>CustomGenerator(Sequence) :

<strong>def          </strong>i n i t         ( self ,        text ,        labels ,               batch size , num steps=None) :

self . text , self . labels = text , labels self . batch size = batchsize

self . <strong>len </strong>= np. ceil (<strong>len</strong>( self . text ) / <strong>float </strong>( self . batch size ) ) . astype (np. int64 ) <strong>if </strong>num steps : self . <strong>len </strong>= <strong>min</strong>(numsteps , self . <strong>len</strong>)

<strong>def          </strong>len         ( self ) :

<strong>return </strong>self . <strong>len</strong>

<strong>def           </strong>getitem          ( self ,       idx ) :

batch x = self . text [ idx ∗ self . batch size &#x1f641; idx + 1) ∗ self . batch size ] batch y = self . labels [ idx ∗ self . batch size &#x1f641; idx + 1) ∗ self . batch size ] <strong>return </strong>batch x ,              batch y

traingen = CustomGenerator( Xtrain ,                Y train ,       batch size ) validgen = CustomGenerator( Xvalid ,                 Y valid ,       batch size ) testgen = CustomGenerator( Xtest ,                Y test ,        batch size )

We feed this data generator and start training the model as shown below:

<em># Training the model </em>checkpointer = ModelCheckpoint( ’data/models/model−{epoch :02d}−{val loss :.5 f }. hdf5 ’ ,

monitor=’ val loss ’ , verbose=1, save best only=True , mode=’min ’ )

callback list = [ checkpointer ] <em>#, , reduce lr </em>his1 = model . fit generator ( generator=train gen , epochs=1,

validation data=valid gen , callbacks=callback list )

<h1>0.4      Evaluating the model</h1>

Once the model is trained, we evaluate it on the test set. predict generator will generate a probability of a given instance to be buggy or clean.

Traditionally, instances will be then classified in the class 0 (i.e., clean) if the probability is lower than 50%, and in the class 1 (i.e., buggy) if the probability is higher. However, using the 50% threshold might not be the best choice and using a different threshold might provide better results. Therefore, to take into consideration the impact of the threshold, we draw the ROC curve and use the AUC (area under the curve metrics) to measure the correctness of our classifier. predIdxs = model . predict generator ( test gen , verbose=1)

fpr , tpr ,      = roc curve ( Y test ,    predIdxs ) roc auc = auc( fpr ,         tpr )

plt . figure ()

lw = 2

plt . plot ( fpr , tpr , color=’ darkorange ’ , lw=lw , label=’ROC curve ( area = %0.2 f ) ’ % roc auc ) plt . plot ([0 , 1] , [0 , 1] , color=’navy ’ , lw=lw , linestyle=’−−’ ) plt . xlim ([0.0 , 1.0]) plt . ylim ([0.0 , 1.05]) plt . xlabel ( ’ False Positive Rate ’ ) plt . ylabel ( ’True Positive Rate ’ ) plt . t i t l e ( ’ Receiver operating characteristic example ’ ) plt . legend ( loc=”lower right”) plt . savefig ( ’auc model . png ’ )

<strong>For Part (I), please include auc model.png in your report, and measure the buggy rate (i.e. % of instances labeled 1) in the training, validation and test instances.</strong>

<h1>(II)- Improving the results by using a better deep-learning algorithm</h1>

The model trained in part (I) is simple and does not perform very well. In the past few years, many different models to classify text inputs for diverse tasks (content tagging, sentiment analysis, translation, etc.) have been proposed in the literature. In part (II), you will look at the literature and apply a different deep-learning algorithm to do defect prediction. You can, and are encouraged to use or adapt models that have been proposed by other people for other tasks. Please cite your source and provide a link to a paper or/and GitHub repository showing that this algorithm has been applied successfully for text classification, modeling or generation tasks.

Examples of models to try:

Hierarchical Attention Networks for Document Classification:

<a href="https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf">https://www.cs.cmu.edu/</a><a href="https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf">~</a><a href="https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf">./hovy/papers/16HLT-hierarchical-attention-networks.pdf</a> and <a href="https://github.com/richliao/textClassifier">https://github.com/ </a><a href="https://github.com/richliao/textClassifier">richliao/textClassifier</a><a href="https://github.com/richliao/textClassifier">,</a>

Independently Recurrent Neural Network (IndRNN): Building A Longer andDeeper RNN: <a href="https://arxiv.org/pdf/1803.04831.pdf">https://arxiv.org/pdf/</a>

<a href="https://arxiv.org/pdf/1803.04831.pdf">1803.04831.pdf</a> and <a href="https://github.com/titu1994/Keras-IndRNN">https://github.com/titu1994/Keras-IndRNN</a>

Feed-Forward Networks with Attention Can Solve Some Long-Term Memory Problems: <a href="https://arxiv.org/pdf/1512.08756.pdf">https://arxiv.org/pdf/1512.</a>

<a href="https://arxiv.org/pdf/1512.08756.pdf">08756.pdf</a> and <a href="https://github.com/ShawnyXiao/TextClassification-Keras">https://github.com/ShawnyXiao/TextClassification-Keras </a>You can also look at more complex models like BERT, Elmo or XLNet.

You can search GitHub for text-classification models and pick the one you like!

<strong>We strongly recommend that you do not implement the CNN-only models from this paper: </strong><a href="https://arxiv.org/abs/1408.5882">https:</a>

<a href="https://arxiv.org/abs/1408.5882">//arxiv.org/abs/1408.5882</a><a href="https://arxiv.org/abs/1408.5882">.</a> We have extensively tested this model for our specific task and we already know it does not work well.

<strong>Report: </strong>For this question, please put in your report the model you chose, a link to the paper and/or GitHub repository where you got the model, a small discussion why you chose to try this model, your source code, and an evaluation of your trained model on the test set (AUC and ROC curve). If you get any improvement compared to the model used in part I, please report it too.

If the model you pick is too complex and takes too long to train on the entire training set, please provide an explanation indicating how much time the model would take to train on the entire dataset and only train your model on a sample of the dataset.

<h1>(III)- Other ways to improve the results</h1>

In this question, you will try to improve the model you worked with in Part II using different methods. <strong>Chose at least two of the methods below </strong>to try to improve the results you got in part II: Report which methods you use and its impact on the results and training time.

<strong>Use more training data: </strong>In part I), we only used 50,000 instances to train our model. You can try to train your model with the entire training set instead. Based on our experience, using 1 million instances produce much higher AUC than using 50,000 instances. Generally, the more training instances, the higher AUC until it saturates. The constraint is machine time.

<strong>Data cleaning: </strong>The input data we provided is automatically extracted from GitHub and likely contains a lot of noise. To improve the results, one possibility is to clean the datasets. You can investigate a bit more the raw data and try to clean the input data. Examples (non-exhaustive) of challenges to investigate and solve are:

<ul>

 <li>Duplicate instances: Are there any instances that are labeled both buggy and clean?</li>

 <li>Length of the input: What is the average length of an instance, are there any outliers? Does removing outliers improve the results?</li>

 <li>Quality of the input: Comments have not been removed from the inputs? Does removing comments help to improve the results?</li>

</ul>

<strong>Tokenization and input abstraction:</strong>

In this project, we use a simple tokenization using a java tokenizer and basic abstraction of strings and numbers. This has the inconvenience of creating a gigantic vocabulary that might be difficult to learn for a deep learning network. Many different tokenizers or abstractions can be tried:

<ul>

 <li>Source code contains structured information that could help abstract data to reduce the vocabulary size. For example, all variables could be abstracted to the same token <em>variable</em>, all method calls to the token <em>method<sub>c</sub>all</em>, types to <em>type</em>, etc. You can also distinguish between different variables in the same instance by abstracting different variables with slightly different tokens (e.g., <em>var</em><sub>1</sub>, <em>var</em><sub>2</sub>, etc. Such information can be extracted from an AST or a java Parser (the javalang library contains a basic AST parser that could be used). Using such an abstract will significantly reduce the vocabulary and might help the algorithm to learn.</li>

 <li>Subword tokenizers have been used in NLP. You can try tokenizers like SentencePiece (<a href="https://github.com/google/sentencepiece">https://github.com/ </a><a href="https://github.com/google/sentencepiece">google/sentencepiece</a><a href="https://github.com/google/sentencepiece">,</a> or word pieces (<a href="https://www.tensorflow.org/datasets/api_docs/python/tfds/features/text/SubwordTextEncoder">https://www.tensorflow.org/datasets/api_docs/python/tfds/feature</a>s/ <a href="https://www.tensorflow.org/datasets/api_docs/python/tfds/features/text/SubwordTextEncoder">text/SubwordTextEncoder</a><a href="https://www.tensorflow.org/datasets/api_docs/python/tfds/features/text/SubwordTextEncoder">)</a></li>

 <li>You can also build your own tokenizer.</li>

</ul>

<strong>Context representation: </strong>In part I), the context is represented as a sequence of tokens from the entire function. In addition, both the context and the line under test are represented similarly and fed as one input. This might not be the best way to represent the context of a bug. You can propose a different approach to represent the context of a bug:

<ul>

 <li>You can try to represent the context differently (e.g., use a higher-level abstraction, only use a set of tokens instead of a sequence)</li>

 <li>In this project, we use the entire function as context. This provides a lot of information, but it likely also contains a lot of noise (e.g. irrelevant statements). You can try to use a different context (e.g., reduce the context to only consider the basic block surrounding the line under test).</li>

 <li>You can try to feed the context and the instance under test as different inputs.</li>

</ul>

<strong>Tuning and Building Deeper models: </strong>Deep learning models contain a lot of hyper-parameters that can be tuned (e.g., number of epoch trained, number and size of layers, dropout rate, learning rate, etc.). Using different hyper-parameters can lead to very different results. One way to improve the results of a classifier is to pick the ”best” hyper-parameters by tuning the model.

<strong>Using different Learning methods: </strong>Sometimes, learning from one model and one dataset is not enough to achieve good results. There are several possibilities to improve the models:

<ul>

 <li>Use pre-trained embedding to have a better source code representation. Much work has been done to represent source code from a very large corpus. Instead of training our embedding layers from our limited training data, you could use a pre-trained embedding (e.g. such as the ones proposed in code2seq <a href="https://github.com/tech-srl/code2seq">https://github.com/tech-srl/ </a><a href="https://github.com/tech-srl/code2seq">code2seq</a> or train your own embedding (e.g., GloVe or Word2Vec) before training the classifier.</li>

 <li>It is easier to learn from simple instances first. Curriculum Learning has been proposed to help to learn easier instances first. (<a href="https://arxiv.org/abs/1904.03626">https://arxiv.org/abs/1904.03626</a><a href="https://arxiv.org/abs/1904.03626">)</a></li>

 <li>Use ensemble learning. One model might not be enough to learn all buggy lines. Instead of building one single model, a combination of several smaller models (trained with different training data or using different hyperparameters) might provide better performances.</li>

</ul>

<h1>(IV)- Further improvements (competition) – for Bonus</h1>

You are also highly encouraged to improve the defect prediction models by using other techniques beyond the ones we recommended or to try to combine all of them to further improve your model.

<h1>(Optional) Use GPU for your training</h1>

GPU can drastically accelerate the speed of training. In this part, we will guide you to use tensorflow-gpu to train the model.

Server cuda.cs.purdue.edu is equipped with 6 GPUs capable of deep learning. You should have access to this server. However, most of the time, its GPUs are occupied by others, which is out of our control. We highly recommend that you consider using GPUs if you have access to one.

<h2>Use GPU of cuda.cs.purdue.edu</h2>

We have environment ready to use on this server. Run nvidia-smi to check the avaibility of GPUs before you start. To run training on this server using GPU, you should follow the steps:

module load cuda/10.0

source /homes/cs510/project −3/venv−gpu/<strong>bin</strong>/activate

python train and test .py

You may get a OUT OF MEMORY error if no GPU is available at that time. Since we don’t have control over the server, we cannot guarantee your access to the GPU. You may try at different time.

<h2>Use your own GPU</h2>

If you have control over a machine with Nvidia GPU. You may use tensorflow-gpu to accelerate your training (The performance is varied based on the model).

<strong>Prerequisites</strong>

<ul>

 <li>Python3</li>

 <li>CUDA Toolkit 10.0 (<a href="https://developer.nvidia.com/cuda-10.0-download-archive">https://developer.nvidia.com/cuda-10.0-download-archive</a><a href="https://developer.nvidia.com/cuda-10.0-download-archive">)</a></li>

 <li>cuDNN (Any version that is compatible with cuda10.0 <a href="https://developer.nvidia.com/cudnn">https://developer.nvidia.com/cudnn</a><a href="https://developer.nvidia.com/cudnn">)</a></li>

</ul>

Once you have meet the prerequisites, you can create a virtualenv and use the provide requirements-gpu.txt to setup your environment.

python3 −m venv path/to/venv

source path/to/venv/<strong>bin</strong>/activate pip install −−upgrade pip pip install −r requirements−gpu . txt

Then, you should be ready to train your model on a large dataset faster.