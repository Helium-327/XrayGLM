#!/usr/bin/env python
"""
Gradio Web Demo for XrayGLM with stop button and layout improvements.
"""

import gradio as gr
from PIL import Image
import os
import json
from model import is_chinese, generate_input, chat
import torch
import argparse
from transformers import AutoTokenizer
from model import VisualGLMModel, chat
from finetune_XrayGLM import FineTuneVisualGLMModel
from sat.model import AutoModel
from sat.model.mixins import CachedAutoregressiveMixin
from sat.quantization.kernels import quantize

def generate_text_with_image(input_text, image, history=[], request_data=dict(), is_zh=True):
    input_para = {
        "max_length": 2048,
        "min_length": 50,
        "temperature": 0.8,
        "top_p": 0.4,
        "top_k": 100,
        "repetition_penalty": 1.2
    }
    input_para.update(request_data)

    input_data = generate_input(input_text, image, history, input_para, image_is_encoded=False)
    input_image, gen_kwargs = input_data['input_image'], input_data['gen_kwargs']
    with torch.no_grad():
        answer, history, _ = chat(None, model, tokenizer, input_text, history=history, image=input_image,
                                  max_length=gen_kwargs['max_length'], top_p=gen_kwargs['top_p'],
                                  top_k=gen_kwargs['top_k'], temperature=gen_kwargs['temperature'], english=not is_zh)
    return answer


def request_model(input_text, temperature, top_p, image_prompt, result_previous):
    result_text = [(ele[0], ele[1]) for ele in result_previous]
    for i in range(len(result_text) - 1, -1, -1):
        if result_text[i][0] == "" or result_text[i][1] == "":
            del result_text[i]
    print(f"history {result_text}")

    is_zh = is_chinese(input_text)
    if image_prompt is None:
        if is_zh:
            result_text.append((input_text, '图片为空！请上传图片并重试。'))
        else:
            result_text.append((input_text, 'Image empty! Please upload a image and retry.'))
        return input_text, result_text
    elif input_text == "":
        result_text.append((input_text, 'Text empty! Please enter text and retry.'))
        return "", result_text

    request_para = {"temperature": temperature, "top_p": top_p}
    image = Image.open(image_prompt)
    try:
        answer = generate_text_with_image(input_text, image, result_text.copy(), request_para, is_zh)
        # answer = "测试回答：这是一个示例答案"  # 硬编码测试值
    except Exception as e:
        print(f"error: {e}")
        if is_zh:
            result_text.append((input_text, '超时！请稍等几分钟再重试。'))
        else:
            result_text.append((input_text, 'Timeout! Please wait a few minutes and retry.'))
        return "", result_text

    print("Input Text:", input_text)  # 确认输入文本
    print("Generated Answer:", answer)  # 确认生成内容

    result_text.append((input_text, answer))
    print(result_text)
    return "", result_text


DESCRIPTION = '''# <a href="https://github.com/WangRongsheng/XrayGLM">XRAY-GLM</a>'''

MAINTENANCE_NOTICE1 = 'Hint 1: If the app report "Something went wrong, connection error out", please turn off your proxy and retry.\nHint 2: If you upload a large size of image like 10MB, it may take some time to upload and process. Please be patient and wait.'
MAINTENANCE_NOTICE2 = '提示1: 如果应用报了“Something went wrong, connection error out”的错误，请关闭代理并重试。\n提示2: 如果你上传了很大的图片，比如10MB大小，那将需要一些时间来上传和处理，请耐心等待。'

NOTES = 'This app is adapted from <a href="https://github.com/WangRongsheng/XrayGLM">https://github.com/WangRongsheng/XrayGLM</a>. It would be recommended to check out the repo if you want to see the detail of our model and training process.'


def clear_fn():
    return "", [("", "Hi, What do you want to know about this image?")], None  # 清空文本、聊天历史、图片

def main(args):
    global model, tokenizer
    
    
    # load model
    model, model_args = AutoModel.from_pretrained(
        args.from_pretrained,
        args=argparse.Namespace(
            fp16=True,
            skip_init=True,
            use_gpu_initialization=True if (torch.cuda.is_available() and args.quant is None) else False,
            device='cuda' if (torch.cuda.is_available() and args.quant is None) else 'cpu',
        ))
    
    # print(model)
    
    model = model.eval()

    if args.quant:
        quantize(model.transformer, args.quant)

    # if torch.cuda.is_available():
    #     model = model.cuda()

    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())

    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)

    with gr.Blocks(css='style.css') as demo:
        gr.Markdown(DESCRIPTION)
        with gr.Row():
            with gr.Column(scale=4, min_width=500):
                # 文本输入框（修正为 Gradio 3.x 组件）
                input_text = gr.Textbox(
                    label='Input Text',
                    placeholder='Please enter text prompt below and press ENTER.',
                    lines=4
                )
                # 图片上传组件（修正为 Gradio 3.x 组件）
                image_prompt = gr.Image(
                    type="filepath",  # 接收文件路径
                    label="Image Prompt",
                    value=None  # 初始值为空
                )
                with gr.Row():
                    run_button = gr.Button("Generate", variant="primary")
                    # stop_button = gr.Button("Stop", variant="secondary")
                    clear_button = gr.Button("Clear", variant="secondary")

                with gr.Group():
                    with gr.Row():
                        # 滑动条组件（修正为 Gradio 3.x 组件）
                        temperature = gr.Slider(
                            maximum=1,
                            value=0.8,
                            minimum=0,
                            label='Temperature'
                        )
                        top_p = gr.Slider(
                            maximum=1,
                            value=0.4,
                            minimum=0,
                            label='Top P'
                        )

                with gr.Group():
                    maintenance_notice = gr.Markdown(MAINTENANCE_NOTICE1)

            with gr.Column(scale=6):
                # 聊天历史组件（修正为 Gradio 3.x 组件）
                result_text = gr.Chatbot(
                    label='Multi-round conversation History',
                    value=[("", "Hi, What do you want to know about this image?")],
                    height=550  # 保留，Gradio 3.x 支持
                )

        gr.Markdown(NOTES)

        # Gradio 3.x 不支持 Cancel 类，停止按钮暂时设为无操作
        # stop_button.click(fn=lambda: None, inputs=[], outputs=[])

        # 绑定生成按钮和文本提交事件
        run_button.click(
            fn=request_model,
            inputs=[input_text, temperature, top_p, image_prompt, result_text],
            outputs=[input_text, result_text]
        )
        input_text.submit(
            fn=request_model,
            inputs=[input_text, temperature, top_p, image_prompt, result_text],
            outputs=[input_text, result_text]
        )

        # 绑定清空按钮事件
        clear_button.click(
            fn=clear_fn,
            inputs=[],  # 无需输入
            outputs=[input_text, result_text, image_prompt]  # 清空三个组件
        )
    demo.launch(share=args.share)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant", choices=[8, 4], type=int, default=None)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--from_pretrained", type=str, default="/mnt/d/MultiModalityMed/XrayGLM/checkpoints", help='pretrained ckpt')
    args = parser.parse_args()
    main(args)
