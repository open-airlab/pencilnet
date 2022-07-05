import os
import json
import shutil
import inspect
import tensorflow as tf
import sys
import json

class Logger:
    
    def __init__(self, name="unknown"):
        self.name = name
        self.config = None
        self.network = None
        self.loss = None
        print("[ INFO ] Logger.__init__: New logger is initiated with name {}".format(name))

    def set_config(self, config):
        self.config = config

    def set_network(self, network):
        self.network = network    

    def set_loss(self, loss):
        self.loss = loss

    def set_save_dir(self, save_dir):
        self.save_dir = save_dir

    def save(self, overwrite=True):
        dest = self.save_dir
        exit = False
        if self.config==None:
            print("[ ERROR ] Logger.save: config has not been set yet. Call set_config(..) before saving.")
            exit = True

        if self.network==None:
            print("[ ERROR ] Logger.save: network has not been set yet. Call set_network(..) before saving.")
            exit = True

        if self.loss==None:
            print("[ ERROR ] Logger.save: loss has not been set yet. Call set_loss(..) before saving.")
            exit = True

        if not os.path.isdir(dest):
            print("[ ERROR ] Logger.save: Destination folder ({}) does not exist. Create directory first: mkdir {}".format(dest, dest))
            exit = True

        if exit:
            quit()

        path = os.path.join(dest, self.name)
        if os.path.isdir(path):
            if overwrite:
                print("[ WARNING ] Logger.save: Log folder ({}) is overwritten.".format(path))
                shutil.rmtree(path)
                os.mkdir(path)
            else:
                print("[ ERROR ] Logger.save: Log folder ({}) exists. Set overwrite True or rename the log folder.".format(path))    
        else:
            os.mkdir(path)

        # Save configs.
        with open(os.path.join(path, "configs.json"), "w+") as outfile:
            json.dump(self.config, outfile, indent=4, default=lambda o: o.__name__)
          
        # Save loss.
        lines = inspect.getsource(self.loss)
        with open(os.path.join(path, "loss.py"), "w+") as outfile:
            outfile.write("# This file is created automatically by a logger. \n")
            outfile.write("import tensorflow as tf\n")
            outfile.write("import numpy as np\n")
            outfile.write("def loss(y_true, y_pred):\n")
            outfile.write(lines[len(lines.split("\n")[0])+1:])

        # Save network.
        lines = inspect.getsource(self.network)
        with open(os.path.join(path, "network.py"), "w+") as outfile:
            outfile.write("# This file is created automatically by a logger. \n")
            outfile.write("import tensorflow as tf\n")
            outfile.write("import numpy as np\n")
            outfile.write("def network_architecture(config):\n")
            outfile.write(lines[len(lines.split("\n")[0])+1:])

        # Save metrics.
        if len(self.config['metrics'])>0:
            with open(os.path.join(path, "metrics.py"), "w+") as outfile:
                outfile.write("# This file is created automatically by a logger. \n")
                outfile.write("import tensorflow as tf\n")
                outfile.write("import numpy as np\n")

            for metric in self.config['metrics']:
                lines = inspect.getsource(metric)
                with open(os.path.join(path, "metrics.py"), "a") as outfile:
                    outfile.write(lines)
                    outfile.write("\n\n")
        model = self.network(self.config)

        # Save model.json config.
        json_config = model.to_json()
        with open(os.path.join(path, 'model.json'), 'w+') as json_file:
            json_file.write(json_config)

        # Save model plot.
        tf.keras.utils.plot_model(model, to_file=(os.path.join(path, 'model.png')), show_shapes=True)

        print("[ INFO ] Logger.save: Log is saved into [{}].".format(path))    

    def get_tensorboard_callback(self):
        """ Tensorbaord callback."""        
        file_writer = tf.summary.create_file_writer(self.config['tensorboard_log_dir'] + "/metrics")
        file_writer.set_as_default()
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=self.config['tensorboard_log_dir'],
        update_freq='batch',
        histogram_freq=1)

        print("[ INFO ] : Logger.get_tensorboard_callback: Tensorboard callback is created in {}".format(self.config['tensorboard_log_dir']))
        return tensorboard_callback

    def get_checkpoint_callback(self):
        # Checkpoint
        save_dir = os.path.join(self.save_dir, self.name, "weights")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filepath=save_dir+"/model_"+self.name+"_-{epoch:02d}.h5"
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath, verbose=1, \
                                    save_best_only=False, save_weights_only=True, \
                                    mode='auto', save_frequency=self.config['save_frequency'])    

        print("[ INFO ] : Logger.get_checkpoint_callback: Checkpoint callback is created for model name {}".format(save_dir+"/model_"+self.name+"_EPOCH.hdf5"))
        return checkpoint_callback                            


    def get_csv_callback(self):
        file_name = os.path.join(self.save_dir, self.name, "history.csv")
        csv_callback = tf.keras.callbacks.CSVLogger(file_name, separator=',', append=False)
        print("[ INFO ] : Logger.get_csv_callback: CSV history file is created at {}".format(file_name))
        return csv_callback


    def load(self, path_to_directory):

            self.path = path_to_directory

            # Load config.
            with open(os.path.join(path_to_directory, "configs.json"), "r") as json_file:
                self.config = json.load(json_file)

            # Load model.
            sys.path.append(path_to_directory)
            from network import network_architecture
            self.model = network_architecture(self.config)

            # Load loss.
            from loss import loss
            self.loss = loss

    # def list_checkpoints(self):
    #     checkpoints_path = os.path.join(self.path, "weights")
    #     onlyfiles = [f for f in os.listdir(checkpoints_path) if os.path.isfile(os.path.join(checkpoints_path, f))]
    #     onlyfiles.sort()
    #     print("[ INFO ] Logger: list_checkpoints: {} checkpoints are found.".format(len(onlyfiles)))
    #     for i in onlyfiles:
    #         print("[ INFO ] Logger: list_checkpoints: {}".format(i)) 


    def list_checkpoints(self):
            checkpoints_path = os.path.join(self.path, "weights")
            onlyfiles = [f for f in os.listdir(checkpoints_path) if os.path.isfile(os.path.join(checkpoints_path, f))]
            onlyfiles = sorted(onlyfiles, key=lambda s: int(s.split('-')[-1].replace('.h5', '')))
            print("[ INFO ] Logger: list_checkpoints: {} checkpoints are found.".format(len(onlyfiles)))
            for i in onlyfiles:
                print("[ INFO ] Logger: list_checkpoints: {}".format(i)) 
            return onlyfiles


    def load_checkpoint(self, epoch, verbose=1):
        weight_path = os.path.join(self.path, "weights", "model_"+self.config['name']+"_-{epoch:02d}.h5".format(epoch=epoch))
        self.model.load_weights(weight_path)
        if verbose: 
            print("[ INFO ]: Logger.load_checkpoint: Checkpoint loaded: {}".format(weight_path))

        return self.model