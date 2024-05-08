from model import *
from utils import *
import rasterio
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_model(reload_path):
    '''
    :param reload_path: path to the model
    return: model
    '''
    input_length_days = 20
    img_size = 64

    InverseQModel = DRQModel(in_channels=8, num_days=input_length_days, img_size=img_size, num_features=384, depth=6)
    state_dict = torch.load(reload_path)

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    InverseQModel.load_state_dict(new_state_dict)
    InverseQModel = InverseQModel.cuda()
    InverseQModel.eval()
    
    return InverseQModel


def handle_nan(arr):
    '''
    :param arr: array
    return: array with nan, inf, -inf values replaced with -1
    '''
    assign = -1
    arr[np.isnan(arr)] = assign
    arr[arr == -float('inf')] = assign
    arr[np.isinf(arr)] = assign
    return arr


def get_prediction(model_input, model, input_days_length, overlapping_days_length, q_mean, transform_pred_func):
    '''
    :param item: model input
    :param model: model
    :param input_days_length: length of the input in days or images
    :param overlapping_days_length: length of the overlapping days
    :param q_mean: long term mean of the discharge
    :param transform_pred_func: function to transform the prediction to the original scale
    return: prediction, prediction in every step
    '''
    if overlapping_days_length >= input_days_length:
        raise ValueError("overlapping_days_length should be less than input_days_length.")

    input = model_input['input'].cuda()
    length = input.shape[1]
    pred_qs = [[] for _ in range(length)]
    prev_pred = None

    step_predictions = []

    step_size = input_days_length - overlapping_days_length

    for i in tqdm(list(range(0, length - input_days_length + 1, step_size))):
        tmp_input = input[:, i:i + input_days_length, :, :, :]
        tmp_pred = model(tmp_input).detach().cpu().numpy().squeeze()
        tmp_pred = transform_pred_func(tmp_pred)

        step_predictions.append(tmp_pred.copy())

        if prev_pred is not None:
            overlap = prev_pred[-overlapping_days_length:]
            scale_ratio = overlap.mean() / tmp_pred[:overlapping_days_length].mean() if tmp_pred[
                                                                                        :overlapping_days_length].mean() != 0 else 1
            tmp_pred *= scale_ratio

        prev_pred = tmp_pred

        for j, pred_q in enumerate(tmp_pred):
            pred_qs[i + j].append(pred_q)

    pred_qs = [np.mean(pred_q) for pred_q in pred_qs]
    pred_qs = np.array(pred_qs)
    pred_qs *= q_mean

    return pred_qs, step_predictions


def transform_pred(pred):
    '''
    :param pred: prediction from the model
    return: transformed prediction in the original scale
    '''
    max_log_of_ratio, min_log_of_ratio = 5.096, -13.815
    pred = np.exp(pred * (max_log_of_ratio - min_log_of_ratio) + min_log_of_ratio)
    return pred


def get_model_input(tif_files, input_size=64):
    '''
    :param tif_files: list of tif files, each file should contain "blue", "green", "red", "nir", "swir1", "swir2", and "qa" bands in that order
    :param input_size: size of the input
    return: model input
    '''
    input_ = []
    for tif_file in tif_files:
        try:
            blue, green, red, nir, swir1, swir2, qa = rasterio.open(tif_file).read()
        except Exception as e:
            print(e)
            print(f'Error reading {tif_file}')
            return
        input_.append({'blue': blue, 'green': green, 'red': red, 'nir': nir, 'swir1': swir1, 'swir2': swir2, 'qa': qa})

    input = []
    for item in input_:
        arr_shape = item['blue'].shape
        center_x_offset = int((arr_shape[0] - input_size) / 2)
        center_y_offset = int((arr_shape[1] - input_size) / 2)

        day_input = []
        for key in ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'srtm', 'awei']:
            if item is None:
                day_input.append(np.zeros((input_size, input_size)))
            else:
                if key == 'srtm':
                    item[key] = np.zeros_like(item['blue'])
                elif key == 'awei':
                    item[key] = awei_index(item['blue'], item['green'], item['nir'], item['swir1'], item['swir2'])
                else:
                    pass
                item[key] = handle_nan(item[key])
                day_input.append(item[key][center_x_offset:center_x_offset + input_size,
                                 center_y_offset:center_y_offset + input_size])
        input.append(day_input)

    input = np.array(input)
    input = handle_nan(input)

    input = torch.from_numpy(input).float()

    assert not torch.isnan(input).any()

    if len(input.shape) == 4:
        input = input.unsqueeze(0)
    return {'input': input}


def process_reach(tif_files, q_mean, model):
    '''
    :param tif_files: list of tif files for a river reach, each file should contain "blue", "green", "red", "nir", "swir1", "swir2", and "qa" bands in that order
    :param q_mean: long term mean of the discharge
    return: predicted discharge in order of the tif files
    '''
    model_input = get_model_input(tif_files)

    overlapping_days_length = 19
    input_days_length = 20
    pred_qs, step_predictions = get_prediction(model_input, model, input_days_length, overlapping_days_length, q_mean,
                                              transform_pred)
    return pred_qs, step_predictions


if __name__ == '__main__':
    model = get_model(reload_path='./DRQ_vCloud_0.01_2024_04_23_17_14.pth')

    dir_path = './Landsat'
    tif_files = absoluteFilePaths(dir_path)
    pred_qs, _ = process_reach(tif_files, q_mean=1.0)
    print(pred_qs)

    plt.plot(pred_qs)
    plt.show()
