from plotting_funcs import *
from build_models import *


if __name__ == '__main__':
    train_generator, validation_generator, holdout_generator = create_data_gens(target_size=(229,229),train_dir = "../../images/Images/train",  val_dir = '../../images/Images/val', holdout_dir = '../../images/Images/test', batch_size = 30)

    X_model = load_model('../../Xception_mod_2.h5')


    Xception_history = X_model.fit(train_generator,
                steps_per_epoch=3034 // 15,
                epochs=10,
                validation_data=validation_generator,
                validation_steps=650 // 16,
                verbose = 1)

    X_model.save('../../Xception_mod3.h5')
