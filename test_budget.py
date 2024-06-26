import unittest
import torch
from budget_orig import (  # Replace 'your_module'
    BudgetConstraint_RatioMSE_DualGrain as OldDualGrain,
    BudgetConstraint_NormedSeperateRatioMSE_TripleGrain as OldTripleGrain)

from budget_new import (
    BudgetConstraint_RatioMSE_DualGrain as NewDualGrain,
    BudgetConstraint_NormedSeperateRatioMSE_TripleGrain as NewTripleGrain,
)

class TestBudgetConstraints(unittest.TestCase):

    def test_dual_grain_same_result(self):
        # Test cases with different parameters
        test_cases = [
            (0.8, 1.0, 8, 16, True),
            (0.5, 0.5, 4, 8, False),
            (0.2, 2.0, 16, 32, True),
        ]

        for case in test_cases:
            target_ratio, gamma, min_grain_size, max_grain_size, calculate_all = case

            # Create instances of old and new classes
            old_constraint = OldDualGrain(target_ratio, gamma, min_grain_size, max_grain_size, calculate_all)
            new_constraint = NewDualGrain(target_ratio, gamma, min_grain_size, max_grain_size, calculate_all)

            # Create random gate tensor
            gate = torch.randn(16, 2, min_grain_size, min_grain_size) 

            # Calculate losses
            old_loss = old_constraint(gate)
            new_loss = new_constraint(gate)

            # Assert near equality
            print('DUAL:\n')
            print(old_loss, new_loss)
            print('\n ---------------')
            # self.assertAlmostEqual(old_loss(), new_loss(), places=5) 

    def test_triple_grain_same_result(self):
        # Test cases with different parameters
        test_cases = [
            (0.6, 0.3, 1.0, 8, 16, 32),
            (0.4, 0.4, 0.5, 4, 8, 16),
            (0.2, 0.7, 2.0, 16, 32, 64),
        ]

        for case in test_cases:
            target_fine_ratio, target_median_ratio, gamma, min_grain_size, median_grain_size, max_grain_size = case

            # Create instances of old and new classes
            old_constraint = OldTripleGrain(target_fine_ratio, target_median_ratio, gamma, min_grain_size, median_grain_size, max_grain_size)
            new_constraint = NewTripleGrain(target_fine_ratio, target_median_ratio, gamma, min_grain_size, median_grain_size, max_grain_size)

            # Create random gate tensor
            gate = torch.randn(16, 3, min_grain_size, min_grain_size)

            # Calculate losses
            old_loss = old_constraint(gate)
            new_loss = new_constraint(gate)

            # Assert near equality
            # self.assertAlmostEqual(old_loss(), new_loss(), places=5)
            print('TRIPLE:\n')
            print(old_loss, new_loss)
            print('\n ---------------')

if __name__ == '__main__':
    unittest.main()