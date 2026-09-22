#pragma once
#include <initializer_list>
#include <vector>
#include <numbers>
#include "classmap.h"

namespace llm
{
	scalar GELU(scalar x);

	class Tensor
	{
		std::vector<size_t> m_Dim;
		std::vector<scalar> m_Data;

	public:
		Tensor(std::initializer_list<size_t> const&);

		//Tensor x{ 2, 3 }
		size_t ndim()const { return m_Dim.size(); }	//   → 2
		size_t size()const { return m_Data.size(); }		//   → 6
		std::vector<size_t> const& shape()const { return m_Dim; }	//  →[2, 3]

		//template<scalar...>
		//scalar at(...)const;		// → 0
		//template<scalar...>
		//scalar& at(...);		// → 0
		scalar at(std::initializer_list<size_t> adr)const { return (*this)[to_index(adr)]; }	// → 0
		scalar& at(std::initializer_list<size_t> adr) { return (*this)[to_index(adr)]; }		// → 0
		scalar operator[](size_t index)const { return m_Data[index]; }
		scalar& operator[](size_t index) { return m_Data[index]; }
		void fill(scalar val) { std::fill(m_Data.begin(), m_Data.end(), val); }
		void set(std::vector<scalar>const& data) { assert(data.size() == size()); m_Data = data; }
		void set(std::vector<scalar>&& data) { assert(data.size() == size()); m_Data = std::move(data); }
		Tensor operator+(Tensor const&)const;
		void operator+=(Tensor const&);
		llm::Tensor gelu()const;

	private:
		size_t to_index(std::initializer_list<size_t> const& adr)const;
	};

	Tensor matmul(const Tensor& left, const Tensor& right);
	Tensor add(const Tensor& left, const Tensor& right);
}