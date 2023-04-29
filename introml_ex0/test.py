import unittest
import numpy as np

from exercise_0 import clip

# set relative tolerance and number of matching decimals
RTOL = 0.01
N_DECIMALS = 2


class TestCalculation(unittest.TestCase):
    def test_clip_min(self):
        res = clip(np.array([5, 4, 3, 2, 1, 2, 3, 4, 5]), 3, 100)
        self.assertEqual(np.min(res), 3)
    
    def test_clip_max(self):
        res = clip(np.array([5, 4, 3, 2, 1, 2, 3, 4, 5]), 0, 4)
        self.assertEqual(np.max(res), 4)
    
    def test_input_preservation(self):
        array = np.array([5, 4, 3, 2, 1, 2, 3, 4, 5])
        res   = clip(array, 0, 4)
        self.assertIsNot(array, res)
    
    def test_length_preservation(self):
        array = np.array([5, 4, 3, 2, 1, 2, 3, 4, 5])
        res   = clip(array, 0, 4)
        self.assertEqual(array.shape, res.shape)
    

if __name__ == '__main__':
    unittest.main()
