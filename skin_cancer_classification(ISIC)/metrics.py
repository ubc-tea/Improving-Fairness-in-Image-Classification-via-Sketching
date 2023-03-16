import numpy as np
from numpy import *

################## Define fairness score ##################
def SPD(prediction,bias,labels, EPS=1e-8): # SPD score -- fair metric one
    acc_privileged = prediction[bias == 1].squeeze().eq(labels[bias == 1]).int().sum().item() / (
                len(labels[bias == 1]) + EPS)
    acc_unprivileged = prediction[bias == 0].squeeze().eq(labels[bias == 0]).int().sum().item() / (
                len(labels[bias == 0]) + EPS)
    score = abs(acc_privileged - acc_unprivileged)
    return score


def TPR0(prediction,bias,labels, EPS=1e-8, nclass=7): #TPR_a=0
    score = 0
    for c in range(nclass):
        pred = prediction[labels == c]
        b = bias[labels == c]
        la = labels[labels == c]
        score += (pred[b == 0].squeeze().eq(la[b == 0]).int().sum().item()+EPS) / (
                    pred.squeeze().eq(la).int().sum().item() + EPS)
    return score/nclass


def TPR1(prediction,bias,labels, EPS=1e-8,nclass=7): #TPR_a=1
    score = 0
    for c in range(nclass):
        pred = prediction[labels == c]
        b = bias[labels == c]
        la = labels[labels == c]
        score += (pred[b == 1].squeeze().eq(la[b == 1]).int().sum().item()+EPS) / (
                    pred.squeeze().eq(la).int().sum().item() + EPS)
    return score/nclass


def FPR0(prediction,bias,labels, EPS=1e-8,nclass=7): #FPR_a=0
    score = 0
    for c in range(nclass):
        pred = prediction[labels == c]
        b = bias[labels == c]
        la = labels[labels == c]
        score += (len(pred[b == 0])- pred[b == 0].squeeze().eq(la[b == 0]).int().sum().item()+EPS) / (
                    len(pred) - pred.squeeze().eq(la).int().sum().item() + EPS)
    return score/nclass


def FPR1(prediction,bias,labels, EPS=1e-8,nclass=7): #FPR_a=1
    score = 0
    for c in range(nclass):
        pred = prediction[labels == c]
        b = bias[labels == c]
        la = labels[labels == c]
        score += (len(pred[b == 1])- pred[b == 1].squeeze().eq(la[b == 1]).int().sum().item()+EPS) / (
                    len(pred) - pred.squeeze().eq(la).int().sum().item() + EPS)
    return score/nclass


def EOD(prediction,bias,labels, EPS=1e-8): # EOD score -- fair metric two
    return abs(TPR0(prediction,bias,labels, EPS) - TPR1(prediction,bias,labels, EPS))


def AOD(prediction,bias,labels, EPS=1e-8): # AOD score -- fair metric three
    return 0.5*(abs(TPR0(prediction,bias,labels, EPS) - TPR1(prediction,bias,labels, EPS)) + abs(FPR0(prediction,bias,labels, EPS) - FPR1(prediction,bias,labels, EPS)))


'''
def TPR0(prediction,bias,labels, EPS=1e-8): #TPR_a=0
    score = prediction[bias == 0].squeeze().eq(labels[bias == 0]).int().sum().item()+EPS / (
                prediction.squeeze().eq(labels).int().sum().item() + EPS)
    return score

def TPR1(prediction,bias,labels, EPS=1e-8): #TPR_a=1
    score = prediction[bias == 1].squeeze().eq(labels[bias == 1]).int().sum().item()+EPS / (
                prediction.squeeze().eq(labels).int().sum().item() + EPS)
    return score

def FPR0(prediction,bias,labels, EPS=1e-8): #FPR_a=0
    score = ~(prediction[bias == 0].squeeze()).eq(labels[bias == 0]).int().sum().item()+EPS / (
                prediction.squeeze().eq(labels).int().sum().item() + EPS)
    return score

def FPR1(prediction,bias,labels, EPS=1e-8): #FPR_a=1
    score = ~(prediction[bias == 1].squeeze()).eq(labels[bias == 1]).int().sum().item()+EPS / (
                prediction.squeeze().eq(labels).int().sum().item() + EPS)
    return score

def EOD(prediction,bias,labels, EPS=1e-8): # EOD score -- fair metric two
    return TPR0(prediction,bias,labels, EPS) - TPR1(prediction,bias,labels, EPS)
'''