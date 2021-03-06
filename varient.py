
"""
Single Nucleotide Polymorphism (SNP) detection for Single Molecule DNA Sequencing
Reassemble sequence from short reads and call variants
Dataset from aggregated alignment BAM file:
    NA12878 PacBio read dataset generated by WUSTL (https://www.ncbi.nlm.nih.gov//bioproject/PRJNA323611)
    Alignment information stored in 3- 15x4 matrices (15 nt, 4 bases coded as one-hot-coding)
    1- baseline encoded refernce sequence and total counts
    2- difference between reference counts and seq counts
    3- difference between reference counts and insertion counts
Align to known variants:
    GIAB project (ftp://ftp-trace.ncbi.nlm.nih.gov:/giab/ftp/release/NA12878_HG001/NISTv3.3.2/GRCh38) 
    List of 15nt sequences containing variant candidates at center postion
    
Train using the high confidence calls on chromosome 21 and test on chromosome 22
Model:
Two convolution + maxpool layers 
Three fully connected layers. 
The output layer contains two group of outputs, each a 4x array. 
 - Base Identity in [A C G T] 
 - Variant type [het, hom, non-var, other-var] 
 
Based on VariantNet: https://github.com/pb-jchin/VariantNET
Modified to run on Clusterone (distributed task with Tensorboard monitoring)
"""

import os
import logging
import tensorflow as tf
import numpy as np
import intervaltree
import random
from clusterone import get_data_path, get_logs_path

from read_data import *
from model import *

# ----- User defined variables -----
CLUSTERONE_USERNAME = "emanrao"

LOCAL_LOG_LOCATION = '~/Desktop/VariantNN/logs'

LOCAL_DATASET_LOCATION = '~/Desktop/VariantNN'
LOCAL_DATASET_NAME = 'data'

# Create logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
    
def main():
    """
    Identify datasets for training and testing
    aln = Alignment data in 15x3x3 matrices for each datapoint
    var = Expected Variants List
    bed = Genomic coordinate information
    """
    
    #Training Data
    train_aln_filename = 'aln_tensor_chr21'
    train_var_filename = 'variants_chr21'
    train_bed_filename = 'CHROM21_v.3.3.2_highconf_noinconsistent.bed'
    
    #Validation Data
    val_aln_filename = 'aln_tensor_chr22'
    val_var_filename = 'variants_chr22'
    val_bed_filename = 'CHROM22_v.3.3.2_highconf_noinconsistent.bed'
    
    # Training Parameters
    batch_size = 500  # Batch size
    num_epochs = 5  # Number epochs
    train_holdout = 0.2  # Portion of training features used for valisation
    learning_rate = 0.005  # Starting learning rate
    steps_per_epoch = 50 # Number of training steps per epoch
    
#----- Begin Main Code    
    
    # Get environment variables
    try:
        job_name = os.environ['JOB_NAME']
        task_index = os.environ['TASK_INDEX']
        ps_hosts = os.environ['PS_HOSTS']
        worker_hosts = os.environ['WORKER_HOSTS']
    except:
        job_name = None
        task_index = 0
        ps_hosts = None
        worker_hosts = None
        
    # Get local file paths
    PATH_TO_LOCAL_LOGS = os.path.expanduser(LOCAL_LOG_LOCATION)
    ROOT_PATH_TO_LOCAL_DATA = os.path.expanduser(LOCAL_DATASET_LOCATION)
   
    # Flags
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    # Flags for environment variables
    flags.DEFINE_string("job_name", job_name,
                        "job name: worker or ps")
    flags.DEFINE_integer("task_index", task_index,
                         "Worker task index, should be >= 0. task_index=0 is "
                         "the chief worker task that performs the variable "
                         "initialization and checkpoint handling")
    flags.DEFINE_string("ps_hosts", ps_hosts,
                        "Comma-separated list of hostname:port pairs")
    flags.DEFINE_string("worker_hosts", worker_hosts,
                        "Comma-separated list of hostname:port pairs")
    
    # Training file flags
    flags.DEFINE_string("train_aln",
                        get_data_path(
                            dataset_name = "emanrao/variantnn-demo",
                            local_root = ROOT_PATH_TO_LOCAL_DATA,
                            local_repo = LOCAL_DATASET_NAME,
                            path = train_aln_filename
                            ),
                        "Path to training dataset (short reads).")
    flags.DEFINE_string("train_var",
                        get_data_path(
                            dataset_name = "emanrao/variantnn-demo",
                            local_root = ROOT_PATH_TO_LOCAL_DATA,
                            local_repo = LOCAL_DATASET_NAME,
                            path = train_var_filename
                            ),
                        "Path to variants for training dataset.")
    flags.DEFINE_string("train_bed",
                        get_data_path(
                            dataset_name = "emanrao/variantnn-demo",
                            local_root = ROOT_PATH_TO_LOCAL_DATA,
                            local_repo = LOCAL_DATASET_NAME,
                            path = train_bed_filename
                            ),
                        "Path to bed file for training dataset.")
    
    flags.DEFINE_string("log_dir",
                         get_logs_path(root=PATH_TO_LOCAL_LOGS),
                         "Path to store logs and checkpoints.")
    
    # Validation file flags
    flags.DEFINE_string("test_aln",
                        get_data_path(
                            dataset_name = "emanrao/variantnn-demo",
                            local_root = ROOT_PATH_TO_LOCAL_DATA,
                            local_repo = LOCAL_DATASET_NAME,
                            path = val_aln_filename
                            ),
                        "Path to testing dataset (short reads).")
    flags.DEFINE_string("test_var",
                        get_data_path(
                            dataset_name = "emanrao/variantnn-demo",
                            local_root = ROOT_PATH_TO_LOCAL_DATA,
                            local_repo = LOCAL_DATASET_NAME,
                            path = val_var_filename
                            ),
                        "Path to variants for testing dataset.")
    flags.DEFINE_string("test_bed",
                        get_data_path(
                            dataset_name = "emanrao/variantnn-demo",
                            local_root = ROOT_PATH_TO_LOCAL_DATA,
                            local_repo = LOCAL_DATASET_NAME,
                            path = val_bed_filename
                            ),
                        "Path to bed file for testing dataset.")

    # Training parameter flags
    flags.DEFINE_integer("batch_size", batch_size,
                        "Batch size [100].")
    flags.DEFINE_integer("num_epochs", num_epochs,
                        "Number epochs [50].")
    flags.DEFINE_float("train_holdout", train_holdout,
                        "Portion of training features withheld from traing and used for validation [0.2].")
    flags.DEFINE_float("learning_rate", learning_rate,
                        "Starting learning rate [0.0005].")
    flags.DEFINE_integer("steps_per_epoch", steps_per_epoch, 
                         "Number of training steps per epoch")

    # Configure Distributed Environment
    def device_and_target():
        # If FLAGS.job_name is not set, we're running single-machine TensorFlow.
        # Don't set a device.
        if FLAGS.job_name is None:
            print("Running single-machine training")
            return (None, "")

        # Otherwise we're running distributed TensorFlow.
        print("Running distributed training")
        if FLAGS.task_index is None or FLAGS.task_index == "":
            raise ValueError("Must specify an explicit `task_index`")
        if FLAGS.ps_hosts is None or FLAGS.ps_hosts == "":
            raise ValueError("Must specify an explicit `ps_hosts`")
        if FLAGS.worker_hosts is None or FLAGS.worker_hosts == "":
            raise ValueError("Must specify an explicit `worker_hosts`")

        cluster_spec = tf.train.ClusterSpec({
                "ps": FLAGS.ps_hosts.split(","),
                "worker": FLAGS.worker_hosts.split(","),
        })
        server = tf.train.Server(
                cluster_spec, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
        if FLAGS.job_name == "ps":
            server.join()

        worker_device = "/job:worker/task:{}".format(FLAGS.task_index)
        # The device setter will automatically place Variables ops on separate
        # parameter servers (ps). The non-Variable ops will be placed on the workers.
        return (
                tf.train.replica_device_setter(
                        worker_device=worker_device,
                        cluster=cluster_spec),
                server.target,
        )

    device, target = device_and_target()

# ----- Read Data  -----   
    # Check Flags
    if FLAGS.log_dir is None or FLAGS.log_dir == "":
        raise ValueError("Must specify an explicit `log_dir`")
    if FLAGS.train_aln is None or FLAGS.train_aln == "":
        raise ValueError("Must specify an explicit `train_aln`")
    if FLAGS.train_var is None or FLAGS.train_var == "":
        raise ValueError("Must specify an explicit `train_var`")
    if FLAGS.train_bed is None or FLAGS.train_bed == "":
        raise ValueError("Must specify an explicit `train_bed`")
    if FLAGS.log_dir is None or FLAGS.log_dir == "":
        raise ValueError("Must specify an explicit `log_dir`")
    if FLAGS.test_aln is None or FLAGS.test_aln == "":
        raise ValueError("Must specify an explicit `test_aln`")
    if FLAGS.test_var is None or FLAGS.test_var == "":
        raise ValueError("Must specify an explicit `test_var`")
    if FLAGS.test_bed is None or FLAGS.test_bed == "":
        raise ValueError("Must specify an explicit `test_bed`")
        
    print('Training alignment file: ', FLAGS.train_aln)
    print('Training variant file: ', FLAGS.train_var)
    print('Training BED file: ', FLAGS.train_bed)

    print('Testing alignment file: ', FLAGS.test_aln)
    print('Testing variant file: ', FLAGS.test_var)
    print('Testing BED file: ', FLAGS.test_bed)
    
    print('Log Files Saved To: ', FLAGS.log_dir)

    # Read in training data
    if FLAGS.task_index == 0:
        print("Looking for training data in %s" % FLAGS.train_aln)        
    Xtrain, Ytrain, pos_train = get_training_array(FLAGS.train_aln, FLAGS.train_var, FLAGS.train_bed)
    # Read in second dataset for final validation (Test)    
    Xtest, Ytest, pos_test = get_training_array(FLAGS.test_aln, FLAGS.test_var, FLAGS.test_bed)
    
    num_train = int(np.round(Xtrain.shape[0] * (1-FLAGS.train_holdout)))
    num_held = int(Xtrain.shape[0]-num_train)
    print('Training on {:d} features'.format(num_train))
    print('Validating on {:d} features (once per epoch)'.format(num_held)) 
    Xval = Xtrain[num_train:]
    Yval = Ytrain[num_train:]
    Xtrain = Xtrain[:num_train]
    Ytrain = Ytrain[:num_train]
    
    num_batches = int(np.floor(Ytrain.shape[0]/FLAGS.batch_size))
    if num_batches==0: # if defined bach size is below dataset, read as1 batch
        num_batches=1
        FLAGS.batch_size = Ytrain.shape[0]

# ----- Define Graph -----

    tf.reset_default_graph()
    with tf.device(device):            
#        X_in = tf.placeholder(tf.float32, [None, 15, 4, 3])
#        Y_out = tf.placeholder(tf.float32, [None, 8])
        global_step = tf.train.get_or_create_global_step()

        # Create Datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((Xtrain, Ytrain))
#        train_dataset = train_dataset.shuffle(buffer_size=10000)
        train_dataset = train_dataset.batch(FLAGS.batch_size)
#        train_dataset = train_dataset.repeat(FLAGS.num_epochs)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((Xval, Yval))
        val_dataset = val_dataset.batch(Yval.shape[0])
#        val_dataset = val_dataset.repeat(FLAGS.num_epochs)

        test_dataset = tf.data.Dataset.from_tensor_slices((Xtest, Ytest))
        test_dataset = test_dataset.batch(FLAGS.batch_size)

        # Create Iterator
        iter = tf.data.Iterator.from_structure(train_dataset.output_types,
                                           train_dataset.output_shapes)
        features, labels = iter.get_next()
        
        # Create initialisation operations
        train_init_op = iter.make_initializer(train_dataset)
        val_init_op = iter.make_initializer(val_dataset)
        test_init_op = iter.make_initializer(test_dataset)
        
        # Apply model
        with tf.name_scope('predictions'):
            predictions = get_model(features, FLAGS)
        with tf.name_scope('loss'):    
            loss = get_loss(predictions,labels)
        tf.summary.scalar('loss', loss)#add to tboard
         
        with tf.name_scope('train'):
            train_step = (
                tf.train.AdamOptimizer(FLAGS.learning_rate)
                .minimize(loss, global_step=global_step)
                )
            
        summ = tf.summary.merge_all()
        writer = tf.summary.FileWriter(FLAGS.log_dir)
        saver = tf.train.Saver()
        
#%% Train Model with periodic validation
    def run_train_epoch(target, FLAGS, epoch_index):
        print('Epoch {:d} Training...'.format(epoch_index))
        i=1
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.steps_per_epoch*epoch_index)] # Increment number of required training steps
        scaffold = tf.train.Scaffold(
                local_init_op=[train_init_op, val_init_op],
                saver=tf.train.Saver(max_to_keep=5)
                )
        v_loss = np.inf
        with tf.train.MonitoredTrainingSession(
                master=target,
                is_chief=(FLAGS.task_index == 0),
                checkpoint_dir=FLAGS.log_dir,
                hooks = hooks,
                scaffold=scaffold
                ) as sess:
            
            writer.add_graph(sess.graph)
            sess.run(train_init_op) # switch to train dataset
            
            while not sess.should_stop():
                
                [current_loss,_,s] = sess.run([loss, train_step, summ])
                iteration = (epoch_index)*FLAGS.steps_per_epoch + i
                print("Iteration {}  Training Loss: {:.4f}".format(iteration,current_loss))
                i += 1
                #writer.add_summary(s, i)
                if i==FLAGS.steps_per_epoch: # validate on last session
                    sess.run(val_init_op) # switch to val dataset
                    while True:
                        try: # run and save validation parameters
                            v_loss = sess.run(loss)
                            print("Epoch {}  Validation Loss: {:.4f}".format(epoch_index, v_loss))
                        except tf.errors.OutOfRangeError:
                            break
        
        chk = tf.train.latest_checkpoint(FLAGS.log_dir)
        return v_loss, chk
     
    validation_loss={} # make dict to keep track of models
    for e in range(1,FLAGS.num_epochs+1):
        v_loss, chkpt = run_train_epoch(target, FLAGS,e)
        validation_loss[chkpt]=v_loss
    
    # Identify best saved model
    all_chkpts = tf.train.get_checkpoint_state(FLAGS.log_dir).all_model_checkpoint_paths
    all_chkpts = [i for i in all_chkpts if i in validation_loss.keys()]
    for each in all_chkpts:
        if validation_loss[each]<v_loss:
            v_loss = validation_loss[each]
            chkpt = each
                
# ----- Test Model on Different Dataset -----  
    with tf.train.MonitoredTrainingSession(
            master=target,
            is_chief=(FLAGS.task_index == 0)
            ) as sess:  
        saver.restore(sess, chkpt)
        sess.run(test_init_op) # initialize to test dataset
        loss = sess.run(loss)
        
    print("Test Set Loss (independent dataset): {:.4f}".format(loss))

    
if __name__ == "__main__":
    main()
    