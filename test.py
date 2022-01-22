"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse
import os
import torch
import cv2

from src.deep_q_network import DeepQNetwork
from src.color_lines import ColorLines


def get_args():
    parser = argparse.ArgumentParser("""Implementation of Deep Q Network to play Color Lines""")
    parser.add_argument("--saved_folder", type=str, default="trained_models")
    parser.add_argument("--fps", type=int, default=3, help="frames per second")
    parser.add_argument("--output", type=str, default="output/color_lines.mp4", help="the path to output video")
    args = parser.parse_args()
    return args


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    env = ColorLines()
    model = DeepQNetwork()
    checkpoint_path = os.path.join(opt.saved_folder, "color_lines.pth")
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("There is no trained weight!")
        exit(0)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    out = cv2.VideoWriter(opt.output, cv2.VideoWriter_fourcc(*"MJPG"), opt.fps, (411, 431))
    while True:
        next_steps = env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        if torch.cuda.is_available():
            next_states = next_states.cuda()
        predictions = model(next_states)[:, 0]
        index = torch.argmax(predictions).item()
        action = next_actions[index]
        _, done = env.step(action)
        frame = env.render()
        out.write(frame)
        if done:
            break


if __name__ == "__main__":
    opt = get_args()
    train(opt)
