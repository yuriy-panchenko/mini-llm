#pragma once
#include <numbers>
#include <vector>
#include <filesystem>

namespace llm
{
	using scalar = float;
	using vdb = std::vector<scalar>;

	inline scalar GELU(scalar x)
	{
		auto const static f{ static_cast<scalar>(sqrt(2. / std::numbers::pi)) };
		return  .5f * x * (1.f + static_cast<scalar>(tanh(f * (x + .044715 * x * x * x))));
	}

	std::string read_all_files(std::filesystem::path const& root);
	std::string read_file(std::filesystem::path const& filename);
}
