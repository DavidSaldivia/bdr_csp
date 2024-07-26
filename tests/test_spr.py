# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 18:13:56 2022

@author: z5158936
"""
import os
import numpy as np
import cantera as ct

from bdr_csp.models.spr import BlackboxModel

def test_model_0D_blackbox():
    temp = 1000.
    flux = 1.0
    temp_amb = 300.
    receiver = BlackboxModel()
    output =  receiver.run_0D_model(temp=temp, flux = flux, temp_amb = temp_amb)

    assert 0.0 < output["eta_rcv"] < 1.0
    assert set(["eta_rcv", "h_rad", "h_conv"]).issubset(output.keys())
    assert not(any(np.isnan(list(output.values()))))
