/**
 * @file methods/ann/loss_functions/multilabel_margin_loss_impl.hpp
 * @author R Aravind
 *
 * Implementation of the Multi-Label Margin Loss function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_MULTILABEL_MARGIN_LOSS_IMPL_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_MULTILABEL_MARGIN_LOSS_IMPL_HPP

// In case it hasn't been included.
#include "multilabel_margin_loss.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. **/ {

template<typename InputDataType, typename OutputDataType>
MultiLabelMarginLoss<InputDataType, OutputDataType>::
MultiLabelMarginLoss(const bool reduction) : reduction(reduction)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType>
typename InputType::elem_type
MultiLabelMarginLoss<InputDataType, OutputDataType>::Forward(
    const InputType& input, const TargetType& target)
{
  typename InputType::elem_type target_idx, input_target, i, j, lossSum = 0;

  for (i = 0; i < input.n_elem; i++)
  {
      target_idx = i;
      input_target = input[target_idx];
  
    if (target[i] == 1)
    {
      for (j = 0; j < input.n_elem; j++)
      {
        if (target[j] == 0)
        {
          typename InputType::elem_type z = 1 - (input_target - input[j]);
          if (z > 0)
            lossSum += z;
        }
      }
    } 
  }
  
  if (reduction)
    return (lossSum);
  
  return (lossSum/ input.n_elem);
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType, typename OutputType>
void MultiLabelMarginLoss<InputDataType, OutputDataType>::Backward(
  const InputType& input,
  const TargetType& target,
  OutputType& output)
{
  // TODO
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void MultiLabelMarginLoss<InputDataType, OutputDataType>::serialize(
  Archive& ar,
  const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(reduction)
}

} // namespace ann
} // namespace mlpack

#endif
