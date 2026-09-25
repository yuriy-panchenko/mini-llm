#pragma once
#include <random>
#include "classmap.h"

namespace llm
{

	class Vector
		:public vdb
	{
	public:
		Vector(size_t elem, scalar val = {})
			:vdb(elem,val)
		{}

		void set(vdb const& data) { assert(data.size() == size()); ((vdb&)*this) = data; }
		void set(vdb&& data) { assert(data.size() == size()); ((vdb&)*this) = std::move(data); }

		Vector gelu()const { auto ret{ *this }; std::transform(ret.begin(), ret.end(), ret.begin(), GELU); return ret; }

		Vector random(std::mt19937 rng, std::uniform_real_distribution<scalar> dist)const
		{
			auto ret{ *this };
			std::transform(ret.begin(), ret.end(), ret.begin(),
				[&](double) { return dist(rng); });
			return ret;
		}
	};
}