import numpy as np

from studio.audio import reference_activity


def test_isolated_boundary_click_does_not_set_the_reference_start():
    rate = 24000
    time = np.arange(4*rate)/rate
    voice = 0.02*np.sin(2*np.pi*180*time)
    signal = np.r_[np.zeros(rate//2), voice, np.zeros(rate//2)]
    signal[100] = 0.9
    active, removed = reference_activity(signal, rate)
    assert len(removed) == 1
    assert np.flatnonzero(active)[0]*0.02 >= 0.48
    assert np.sum(active)*0.02 >= 3.9


def test_initial_speech_and_short_phones_within_it_are_preserved():
    rate = 24000
    time = np.arange(3*rate)/rate
    voice = 0.1*np.sin(2*np.pi*180*time)
    active, removed = reference_activity(voice, rate)
    assert not removed and active[0] and active[-1]
