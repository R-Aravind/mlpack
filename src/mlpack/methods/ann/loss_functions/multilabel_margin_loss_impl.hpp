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
  // TODO
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
