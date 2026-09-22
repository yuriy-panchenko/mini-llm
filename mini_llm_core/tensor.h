#pragma once
#include <initializer_list>
#include <memory>
#include <vector>
#include "classmap.h"

namespace llm
{
	class Tensor
	{
		std::unique_ptr<scalar[]> m_Data;
		std::vector<size_t> m_Dim;

	public:
		Tensor(std::initializer_list<scalar> const&) {}
	};
}