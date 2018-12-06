import os
from data import CreateDataLoader
from cycle_gan_model import CycleGANModel
from util.visualizer import save_images
from util import html
import config


if __name__ == '__main__':
    # Testing
    config.ntest = 2000
    config.results_dir = '../test_results/'
    config.aspect_ratio = 1.0
    config.eval = True
    config.num_test = 50
    config.phase = 'test'
    config.model = 'test'
    config.loadSize = config.fineSize

    config.num_threads = 1   # test code only supports num_threads = 1
    config.batch_size = 1    # test code only supports batch_size = 1
    config.serial_batches = True  # no shuffle
    config.no_flip = True    # no flip
    config.display_id = -1   # no visdom display

    model = CycleGANModel()
    model.initialize(config)
    model.setup(config)
    print("Network Model Created")
    data_loader = CreateDataLoader(config)
    dataset = data_loader.load_data()


    # create a website
    web_dir = os.path.join(config.results_dir, config.name, '%s_%s' % (config.phase, config.epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (config.name, config.phase, config.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # CycleGAN: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    # model.eval()
    for i, data in enumerate(dataset):
        if i >= config.num_test:
            break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        if i % 5 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=config.aspect_ratio, width=config.display_winsize)
    # save the website
    webpage.save()
