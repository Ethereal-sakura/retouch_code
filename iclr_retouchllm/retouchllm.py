import os
import copy
import json
import base64
import io

import torch
from PIL import Image
import argparse
from openai import OpenAI

from code_tools import *
from diff_tools import *
from utils import *
from metrics import *

# OpenAI 格式 API 配置
API_BASE_URL = "http://localhost:10087/v1"
API_KEY = "123"
API_MODEL = "InternVL3-14B"


def pil_to_base64(img, format="JPEG"):
    """将 PIL Image 转为 base64 字符串"""
    buffered = io.BytesIO()
    img.save(buffered, format=format, quality=95)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def history_to_openai_format(history):
    """将内部 chat history 转为 OpenAI API 格式"""
    api_messages = []
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, str):
            api_messages.append({"role": role, "content": content})
        else:
            # content 为 list，可能包含 text 和 image
            api_content = []
            for item in content:
                if item["type"] == "text":
                    api_content.append({"type": "text", "text": item["text"]})
                elif item["type"] == "image":
                    b64 = pil_to_base64(item["image"])
                    api_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    })
            # 若仅含一个 text，部分 API 支持简化为字符串
            if len(api_content) == 1 and api_content[0]["type"] == "text":
                api_messages.append({"role": role, "content": api_content[0]["text"]})
            else:
                api_messages.append({"role": role, "content": api_content})
    return api_messages


def chat_with_api(messages, max_tokens=8192):
    """使用 OpenAI 格式 API 进行对话（支持多模态）"""
    api_messages = history_to_openai_format(messages)
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    response = client.chat.completions.create(
        model=API_MODEL,
        messages=api_messages,
        max_tokens=max_tokens,
        temperature=0.7,
    )
    return response.choices[0].message.content

# VLM prompts
def diff_systemprompt():
    DIFF_SYSTEMPROMPT = f'''
    Task:
    You are an advanced image analysis assistant. Multiple images will be provided along with their color statistics. The first image is the source image, and the rest of the images are the target images. The content and the photometric style of the source and target images differ. The photometric styles of all the target images are the same.
    Your task is to compare the source and target images in terms of the photometric style and identify how the target images differ from the source image in the specific photometric aspects: Exposure, Contrast, Highlight, Shadow, Saturation, Temperature, Texture.

    Instructions:
    1. Choose whether to increase, decrease, or maintain the factor for each aspect. If adjusting, select the appropriate adjustment range from the given options, and if maintaining, write 'N/A' for that aspect.
    2. If adjustments are needed for one or more aspects, write 'go' for the Overall part, while no adjustments are needed for any aspect, write 'stop'.

    Output Format:
    - Exposure: [description of exposure difference, e.g., the brightness of the target image is 10-20% higher than the one of the source image. or N/A.]
    - Contrast: [description of contrast difference, e.g., the contrast of the target image is 10-20% higher than the one of the source image. or N/A.]
    - Highlight: [description of highlight difference, e.g., the highlight of the target image is 10-20% higher than the one of the source image. or N/A.]
    - Shadow: [description of shadow difference, e.g., the shadow of the target image is 10-20% higher than the one of the source image. or N/A.]
    - Saturation: [description of saturation difference, e.g., the saturation of the target image is 10-20% higher than the one of the source image. or N/A.]
    - Temperature: [description of temperature difference, e.g., the temperature of the target image is 10-20% higher than the one of the source image. or N/A.]
    - Texture: [description of texture difference, e.g., the texture of the target image is 10-20% higher than the one of the source image. or N/A.]
    - Overall: Write 'Stop' if there is an N/A for all aspects, and 'Go' if one or more aspects have differences.
    '''
    return DIFF_SYSTEMPROMPT

def diff_userprompt(stat_list: list, range_list: list, score_dict: dict, num_candidate: int = 3):
    def dict2text(info_dict):
        out_text = ''
        for name, value in info_dict.items():
            if isinstance(value, list):
                out_text += f'{name}: {value}, '
            else:
                value = round(value, 2)
                out_text += f'{name}: {value:.2f}, '
        return out_text[:-2]

    def score2text(score_dict):
        out_text = ''
        for name, value in score_dict.items():
            if name in ['psnr', 'delta_e']:
                value = round(value, 2)
                out_text += f'{name}: {value:.2f}, '
            else:
                value = round(value, 3)
                out_text += f'{name}: {value:.3f}, '
        return out_text[:-2]

    if num_candidate == 3:
        DIFF_USERPROMPT_INIT = f'''
        Task:
        You should generate 3 candidate descriptions. Each candidate should include the difference of all the aspects.
        Compare the source image and the target images in terms of the photometric adjustments made to the image, and describe the difference in each aspect.
        You can choose the range from the following list: {range_list}%. Do not exceed the range.
        You can use the color statistics and the scores between source and target image as a guide.

        Color Statistics:
        - Source: {dict2text(stat_list[0])}.
        - Targets (averaged): {dict2text(stat_list[1])}

        Averaged scores (PSNR, SSIM, LPIPS, Delta E):
        {score2text(score_dict)}

        Output Format:
        Candidate 1
        [Description of the first candidate]

        Candidate 2
        [Description of the second candidate]

        Candidate 3
        [Description of the third candidate]
        '''
    return DIFF_USERPROMPT_INIT

# LLM prompts
def code_systemprompt():
    CODE_SYSTEMPROMPT = f'''
    Task:
    You are an expert Python programmer.
    Your task is to generate Python code that sets the appropriate filters and parameter values based on the given photometric aspect-wise description of the color tone difference between the source image and the target image, and arranges the sequence of those steps to make the source image resemble the target image.

    Based on the given description, choose one of the following three options and proceed with the corresponding photometric adjustments:
    - Global Brightness Adjustment (exposure, contrast): If global brightness adjustments are needed more than 1%, focus on modifying elements that affect overall brightness. Do not adjust local brightness, color tone, and texture elements at this stage, only global brightness-related factors.
    - Local Brightness Adjustment (highlight, shadow): If the global brightness adjustments are completed with less than 1% differences, focus on modifying elements that affect local brightness. Do not adjust global brightness, color tone, and texture elements at this stage, only local brightness-related factors.
    - Color Tone and Texture Adjustment (saturation, temperature, texture): If both the global and local brightness adjustments are completed with less than 1% differences, focus on modifying elements that affect color tone and texture. Do not adjust global brightness and local brightness elements at this stage, only color tone and texture-related factors.
    '''
    return CODE_SYSTEMPROMPT

# 4. The adjusted image with save_pth="{save_adj_img_name}".
def code_userprompt(save_adj_img_name, diff):
    CODE_USERPROMPT = f'''
    Instructions:
    1. Examine the given photometric difference description to determine which option to choose, and select only one option from the three options. Ensure that no other options are executed in the code.
    2. Select the appropriate filters for the selected adjustment option, and arrange filters in the correct order.
    3. The filter parameters can be chosen randomly within the range specified in the description.
    4. The variable name of the adjusted image is "{save_adj_img_name}".

    Difference Description:
    {diff}.

    Available Functions:
    - "filter.exposure(f_exp: float) -> np.ndarray": Adjusts the exposure (overall brightness) of an image. The f_exp parameter is an exposure adjustment factor, ranging from -1 to 1. The positive f_exp values brighten the overall image, while negative values darken it.
    - "filter.contrast(f_cont: float) -> np.ndarray": Adjusts the contrast of an image by scaling its pixel values relative to the mean brightness of the image. The f_cont parameter is a contrast adjustment factor, ranging from -1 to 1. Positive f_cont values increase the contrast, making the image more vivid but potentially losing detail in bright and dark areas, while negative values reduce the contrast, retaining more detail but making the image look softer.
    - "filter.highlight(f_high: float) -> np.ndarray": Adjusts the brightness of the bright areas of an image. The f_high parameter is a highlight adjustment factor, ranging from -1 to 1. The positive f_high vlaues intensify the highlights, and negative values reduce them to recover details.
    - "filter.shadow(f_shad: float) -> np.ndarray": Adjusts the brightness of the dark areas of an image.  The f_shad parameter is a shadow adjustment factor, ranging from -1 to 1. The positive f_shad values brighten the shadows and negative values deepen them.
    - "filter.saturation(f_sat: float) -> np.ndarray": Adjusts the saturation of an image. The f_sat parameter is a saturation adjustment factor, ranging from -1 to 1. The positive f_sat values increase color vibrancy, while negative values desaturate the image towards grayscale.
    - "filter.temperature(f_temp: float) -> np.ndarray": Adjusts the color temperature of an image by modifying the balance between warm and cool tones in the RGB color space. The f_temp parameter is a temperature adjustment factor, ranging from -1 to 1. The positive f_temp values shift colors toward warmer tones by increasing red, while negative values shift colors toward cooler tones by enhancing blue.
    - "filter.texture(f_text: float) -> np.ndarray": Adjusts the texture of an image by modifying its high-frequency details using Gaussian blur. The f_text parameter is a texture adjustment parameter, ranging from -1 to 1. The positive f_text values enhance texture by amplifying high-frequency details, while negative values soften texture.

    Please return the code directly without any imports or additional explanations.
    Ensure the code is clear, correct, and follows the steps logically.
    '''
    return CODE_USERPROMPT

# modules
def get_diff(src_img, tar_imgs, diff_history, range_list, score_size, num_candidate):
    # get user prompt
    src_stat = get_stat(src_img, load=False)
    tar_stats = [get_stat(tar_img, load=False) for tar_img in tar_imgs]
    mean_tar_stats = {key: None for key in tar_stats[0]}

    list_format = ['rgb mean', 'hsv mean', 'lab mean']
    for key in mean_tar_stats.keys():
        if key in list_format:
            mean_tar_stats[key] = [
            round(sum(d[key][i] for d in tar_stats) / len(tar_stats), 2)
            for i in range(len(tar_stats[0][key]))]
        else:
            mean_tar_stats[key] = round(sum(d[key] for d in tar_stats) / len(tar_stats), 2)
    stat_list = [src_stat, mean_tar_stats]

    tar_img_list = []
    score_dict = {'psnr': [], 'ssim': [], 'lpips': [], 'delta_e': []}
    for tar_img in tar_imgs:
        (psnr, ssim, lpips, delta_e) = get_final_scores(src_img, tar_img, score_size, p=False)

        tar_img_list.append(tar_img)
        score_dict['psnr'].append(psnr)
        score_dict['ssim'].append(ssim)
        score_dict['lpips'].append(lpips)
        score_dict['delta_e'].append(delta_e)

    averaged_score_dict = {key: sum(value) / len(value) for key, value in score_dict.items()}
    images_list = [src_img] + tar_img_list
    DIFF_USERPROMPT = diff_userprompt(stat_list, range_list, averaged_score_dict, num_candidate)

    diff_history += [{'role': 'user', 'content': [{'type': 'text', 'text': DIFF_USERPROMPT}]}]
    diff_his = copy.deepcopy(diff_history)
    for img in images_list:
        diff_his[-1]['content'].append({'type': 'image', 'image': img})

    out = chat_with_api(diff_his, max_tokens=8192)
    return out, stat_list, diff_history

def get_code(diff, code_history, save_adj_img_name):
    # get user prompt
    CODE_USERPROMPT = code_userprompt(save_adj_img_name, diff)
    code_history += [{'role': 'user', 'content': [{'type': 'text', 'text': CODE_USERPROMPT}]}]

    out = chat_with_api(code_history, max_tokens=8192)
    code_history += [{'role': 'assistant', 'content': [{'type': 'text', 'text': out}]}]
    return code_history

def try_exec(code, diff, code_history, save_adj_img_name, n_iter):
    try:
        exec(code)
    except Exception as e:
        print(f"Error occurred: {n_iter} \n{code}")
        # If exec fails, regenerate the code and try again
        code_history_ = get_code(diff, code_history, save_adj_img_name)
        code_re = code_history_[-1]['content'][-1]['text'].replace('```', '').replace('python\n', '')
        print(f"Re-generated code:\n{code_re}")
        try:
            exec(code_re)
        except Exception as e:
            print(f"Error occurred: {n_iter} \n{code}")
            # If exec fails, regenerate the code and try again
            code_history_ = get_code(diff, code_history, save_adj_img_name)
            code_re = code_history_[-1]['content'][-1]['text'].replace('```', '').replace('python\n', '')
            print(f"Re-generated code:\n{code_re}")
            try:
                exec(code_re)
            except Exception as e:
                print("Fail")
                overall_history['re_list'].append(img_name)

def get_code_from_stat(src_img, tar_imgs):
    src_stat = get_stat(src_img, load=False)
    tar_stats = [get_stat(tar_img, load=False) for tar_img in tar_imgs]
    mean_tar_stats = {key: None for key in tar_stats[0]}

    list_format = ['rgb mean', 'hsv mean', 'lab mean']
    for key in mean_tar_stats.keys():
        if key in list_format:
            mean_tar_stats[key] = [
            round(sum(d[key][i] for d in tar_stats) / len(tar_stats), 2)
            for i in range(len(tar_stats[0][key]))]
        else:
            mean_tar_stats[key] = round(sum(d[key] for d in tar_stats) / len(tar_stats), 2)

    f_exp = round((float(mean_tar_stats['pixel mean']) - float(src_stat['pixel mean'])) / 255 , 2)
    f_cont = round((float(mean_tar_stats['pixel std']) - float(src_stat['pixel std'])) / 127.5 , 2)
    f_high = round((float(mean_tar_stats['pixel percentail 90%']) - float(src_stat['pixel percentail 90%'])) / 255 , 2)
    f_shad = round((float(mean_tar_stats['pixel percentail 10%']) - float(src_stat['pixel percentail 10%'])) / 255 , 2)
    f_sat = round((float(mean_tar_stats['saturation mean']) - float(src_stat['saturation mean'])) / 255 , 2)
    f_temp = round((float(mean_tar_stats['b-channel mean']) - float(src_stat['b-channel mean'])) / 255 , 2)
    f_tex = round((float(mean_tar_stats['laplacian variance']) - float(src_stat['laplacian variance'])) / 255 , 2)

    code_final = ''
    code_final += f"img = filter.exposure({f_exp})\n"
    code_final += f"img = filter.contrast({f_cont})\n"
    code_final += f"img = filter.highlight({f_high})\n"
    code_final += f"img = filter.shadow({f_shad})\n"
    code_final += f"img = filter.saturation({f_sat})\n"
    code_final += f"img = filter.temperature({f_temp})\n"
    code_final += f"img = filter.texture({f_tex})\n"
    return code_final

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, help='The image name', default='a1535-kme_501')
    parser.add_argument('--root_dir', type=str, help='The root dir of the retouching folder')
    args = parser.parse_args()

    model_name = API_MODEL
    print(f'using API: {API_BASE_URL}, model: {model_name}')

    num_iter = 10
    num_candidate = 3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'num candidate: {num_candidate}')
    print(f'image: {args.img}')
    print(device)

    with open(f'{args.root_dir}/adobe_test_500_info_new.json', 'r') as file:
        test_info = json.load(file)
    img_info_dict = test_info['img_info']

    img_name = args.img
    ori_img_pth = f'./samples/src_{img_name}.tif'
    ori_img = Image.open(ori_img_pth).convert('RGB')

    gt_img_pth = f'./samples/gt_{img_name}.tif'
    gt_img = Image.open(gt_img_pth).convert('RGB')

    # tar_names = [
    #             "a1665-jn_20080821_NYC_006",
    #             "a2647-jmac_DSC1283",
    #             "a3351-KE_-1722",
    #             "a3661-WP_CRW_0405",
    #             "a3886-_DGW6415"
    #         ]
    tar_names = ["a1535-kme_501"]
    tar_img_pths = [f'./samples/{tar_name}.tif' for tar_name in tar_names]
    tar_imgs = [Image.open(tar_img_pth).convert('RGB') for tar_img_pth in tar_img_pths]

    src_size = (img_info_dict[img_name]['short256'])
    tar_sizes = [img_info_dict[tar_name]['short256'] for tar_name in tar_names]
    img_size_long500 = img_info_dict[img_name]['long500']

    save_dir_name = f'{args.root_dir}/exp_results/{model_name}/{img_name}'
    os.makedirs(save_dir_name, exist_ok=True)

    DIFF_SYSTEMPROMPT = diff_systemprompt()
    CODE_SYSTEMPROMPT = code_systemprompt()

    overall_history = {'target': gt_img_pth, 'diff': [], 'code': [], 'stat': [], 'r': -1, 'init_score': None, 'final_score': None, 're_list': []} # r: iteration from which the final adjusted image was obtained
    overall_history['init_score'] = get_final_scores(ori_img, gt_img, size=img_size_long500, device=device)

    filter = AdjustmentFilter(ori_img)
    code = get_code_from_stat(ori_img, tar_imgs)
    # code execution
    exec(code)
    warm_img = convert_to_pil(filter.clip_img)
    a = get_final_scores(warm_img, gt_img, size=img_size_long500, device=device)

    num_ori = []
    for n_iter in range(num_iter):
        # 连续3次是一样的选择，直接跳出循环
        if len(num_ori)>2 and num_ori[-3:] == [0, 0, 0]:
            filter.save_img(globals()[f'adjusted_img_{n_iter-1}'], f"{save_dir_name}/adjusted_img.tif")
            overall_history['stat'].append(stat_list)
            overall_history['final_score'] = get_final_scores(globals()[f'adjusted_img_{n_iter-1}'], gt_img, size=img_size_long500, device=device)
            break
        # set system prompt and initial user prompt
        diff_history = [{'role': 'system', 'content': [{'type':'text','text': DIFF_SYSTEMPROMPT}]}]
        code_history = [{'role': 'system', 'content': [{'type':'text','text': CODE_SYSTEMPROMPT}]}]
        if n_iter == 0:
            # get initial difference
            range_list = [(0,1), (1,3), (3,5), (5,10)]
            diff_init, stat_list, diff_history = get_diff(ori_img, tar_imgs, diff_history, range_list, (128, 128), num_candidate)

            overall_history['diff'].append(diff_init)
            diff_history += [{'role': 'assistant', 'content': [{'type': 'text', 'text': diff_init}]}]

            # check whether a difference exists between adjusted and target images
            if diff_init.lower().count('stop') == 1:
                # save history
                ori_img.save(f"{save_dir_name}/adjusted_img.tif")
                overall_history['stat'].append(stat_list)
                overall_history['final_score'] = get_final_scores(ori_img, gt_img, size=img_size_long500, device=device)
                break
            else:
                # filter initialization
                filter = AdjustmentFilter(ori_img, gt_img)

                diff_list = diff_init.lower().split('candidate')[1:]
                diff_comb = [combo.strip()[4:] for combo in diff_list]
                code_init = []
                for i, diff_ in enumerate(diff_comb):
                    adj_img_name = f'adjusted_img_{n_iter}{i}'
                    code_history = get_code(diff_, code_history, adj_img_name)
                    code_i = code_history[-1]['content'][-1]['text'].replace('```','').replace('python\n','')
                    # code execution
                    filter.refresh(ori_img)
                    try_exec(code_i, diff_init, code_history, adj_img_name, n_iter)
                    globals()[adj_img_name] = convert_to_pil(filter.clip_img)
                    code_init.append(code_i)

                src_imgs = [globals()[f'adjusted_img_{n_iter}{i}'] for i in range(num_candidate)]
                src_imgs = [ori_img, warm_img] + src_imgs
                idx = get_idx(src_imgs, tar_imgs, device, s_type='clip', size=img_size_long500)
                src_img = src_imgs[idx]
                num_ori.append(idx)

                if idx == 0:
                    assert src_img == ori_img
                    globals()[f'adjusted_img_{n_iter}'] = src_img
                    code_out = ''
                elif idx == 1:
                    assert src_img == warm_img
                    globals()[f'adjusted_img_{n_iter}'] = src_img
                    code_out = code
                else:
                    globals()[f'adjusted_img_{n_iter}'] = src_img
                    code_out = code_init[idx-2]

                overall_history['code'].append(code_out)
                overall_history['stat'].append(stat_list)
                overall_history['r'] = n_iter
        else:
            ################# n-th round #################
            # get the difference
            range_list = [(0,1), (1,3), (3,5), (5,10)]
            diff, stat_list, diff_history = get_diff(globals()[f'adjusted_img_{n_iter-1}'], tar_imgs, diff_history, range_list, (128, 128), num_candidate)
            diff_history += [{'role': 'assistant', 'content': [{'type': 'text', 'text': diff}]}]
            overall_history['diff'].append(diff)

            if diff.lower().count('stop') == 1:
                # save history
                filter.save_img(globals()[f'adjusted_img_{n_iter-1}'], f"{save_dir_name}/adjusted_img.tif")
                overall_history['stat'].append(stat_list)
                overall_history['r'] = n_iter - 1
                overall_history['final_score'] = get_final_scores(globals()[f'adjusted_img_{n_iter-1}'], gt_img, size=img_size_long500, device=device)
                break
            else:
                # if there still exist difference, adit code based on the difference history
                diff_list = diff.lower().split('candidate')[1:]
                diff_comb = [combo.strip()[4:] for combo in diff_list]
                nth_code = []
                for i, diff_ in enumerate(diff_comb):
                    adj_img_name = f'adjusted_img_{n_iter}{i}'
                    code_history = get_code(diff_, code_history, adj_img_name)
                    code_i = code_history[-1]['content'][-1]['text'].replace('```','').replace('python\n','')
                    # code execution
                    # filter中的图片刷新成上一轮的带的图片
                    filter.refresh(globals()[f'adjusted_img_{n_iter-1}'])
                    try_exec(code_i, diff, code_history, adj_img_name, n_iter)
                    globals()[adj_img_name] = convert_to_pil(filter.clip_img)
                    nth_code.append(code_i)

                src_imgs = [globals()[f'adjusted_img_{n_iter}{i}'] for i in range(num_candidate)]
                src_imgs = [globals()[f'adjusted_img_{n_iter-1}']] + src_imgs
                idx = get_idx(src_imgs, tar_imgs, device, s_type='clip', size=img_size_long500)
                src_img = src_imgs[idx]
                num_ori.append(idx)
                if idx == 0:
                    assert src_img == globals()[f'adjusted_img_{n_iter-1}']
                    globals()[f'adjusted_img_{n_iter}'] = src_img
                    code_out = ''
                else:
                    globals()[f'adjusted_img_{n_iter}'] = src_img
                    code_out = nth_code[idx-1]

                overall_history['code'].append(code_out)
                overall_history['stat'].append(stat_list)
                overall_history['r'] = n_iter

                with open(f'{save_dir_name}/adj_his.json', 'a') as file:
                    file.write(f'R{n_iter}: {stat_list} \n {nth_code}\n\n')

                # 最后一轮进行保存
                if n_iter == num_iter-1:
                    # save history
                    filter.save_img(globals()[f'adjusted_img_{n_iter}'], f"{save_dir_name}/adjusted_img.tif")
                    overall_history['stat'].append(stat_list)
                    overall_history['final_score'] = get_final_scores(globals()[f'adjusted_img_{n_iter}'], gt_img, size=img_size_long500, device=device)
                    break

    with open(f'{save_dir_name}/adj_his.json', 'w') as file:
        json.dump(overall_history, file, cls=NumpyEncoder, indent=2)

