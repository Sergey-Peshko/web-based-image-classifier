import sys
import tensorflow as tf

from DataSetEpochManager import DataSetEpochManager
from DataSetAccuracyManager import DataSetAccuracyManager


def show_progress(epoch, train_acc, val_acc, val_loss, report):
    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, train_acc, val_acc, val_loss))
    report_msg = "{0}\t{1:.3f}\t{2:.3f}\t{3:.3f}\n"
    report.write(report_msg.format(epoch + 1, train_acc, val_acc, val_loss))


def train(epoch_amount, data, session, x, y_true, batch_size, accuracy, cost, optimizer, report_file_name):
    report = open("./train_info/{0}.txt".format(report_file_name), 'w')
    report.write("Training Epoch --- Training Accuracy --- Validation Accuracy --- Validation Loss\n")

    saver = tf.train.Saver()

    passed_epoch = 0

    dataset_epoch_manager = DataSetEpochManager(data.train)

    train_data_accuracy_manager = DataSetAccuracyManager(data.train, session, x, y_true, 400)
    valid_data_accuracy_manager = DataSetAccuracyManager(data.valid, session, x, y_true, 400)

    min_loss = sys.float_info.max

    while passed_epoch < epoch_amount:
        x_batch, y_true_batch = dataset_epoch_manager.next_batch(batch_size)

        feed_dict_tr = {x: x_batch,
                        y_true: y_true_batch}

        session.run(optimizer, feed_dict=feed_dict_tr)

        if dataset_epoch_manager.epochs_done == (passed_epoch + 1):
            # next epoch is come

            train_acc = train_data_accuracy_manager.calc_accuracy(accuracy)
            val_acc = valid_data_accuracy_manager.calc_accuracy(accuracy)
            val_loss = valid_data_accuracy_manager.calc_loss(cost)

            if val_loss <= min_loss:
                min_loss = val_loss
                saver.save(session, '../trained_model/model')

            show_progress(passed_epoch, train_acc, val_acc, val_loss, report)

            passed_epoch += 1

    report.close()
