
from build_models import *


def get_preds(model, holdout_generator):
    pred = model.predict(holdout_generator,verbose=1)
    return pred

def get_real_pred(predictions, holdout_generator):
    predicted_class_indices = np.argmax(pred,axis=1)
    labels = holdout_generator.class_indices
    labels = dict((v,k) for k,v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]
    real_classes = holdout_generator.classes
    real_labels = [labels[k] for k in real_classes]
    return predictions, real_labels


if __name__ == '__main__':
    
    train_generator, validation_generator, holdout_generator = create_data_gens(target_size = (229,229) , train_dir = '../../images/Images/train', val_dir = '../../images/Images/val', holdout_dir =  '../../images/Images/test',  batch_size = 16)

    model = load_final_model()
    pred = get_preds(model, holdout_generator)

    predictions, real_labels = get_real_pred(pred, holdout_generator)



