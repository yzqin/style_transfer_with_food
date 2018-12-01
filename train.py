import time
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
import config

if __name__ == '__main__':

    data_loader = CreateDataLoader(config)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images size= %d' % dataset_size)

    model = create_model(config)
    model.setup(config)
    visualizer = Visualizer(config)
    total_steps = 0

    for epoch in range(config.epoch_count, config.niter + config.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % config.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += config.batch_size
            epoch_iter += config.batch_size
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % config.display_freq == 0:
                save_result = total_steps % config.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % config.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / config.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                if config.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, config, losses)

            if total_steps % config.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                save_suffix = 'iter_%d' % total_steps if config.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % config.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, config.niter + config.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
