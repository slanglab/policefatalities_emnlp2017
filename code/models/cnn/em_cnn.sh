#!/bin/bash
THEANO_FLAGS=mode=FAST_RUN,device=gpu$1,assert_no_cpu_op=raise,floatX=float32 python conv_net_sentence.py 'em' $1 1>&2
