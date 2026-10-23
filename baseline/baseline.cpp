// baseline: n-gram reference losses for the validation set of a mini-llm run.
//
//   baseline <corpus> <vocab.bin> [ctx=64]
//
// Reproduces the split and the validation windows of mini-llm-mfc (CWorkerThread::Init/EvalVal):
// ids = encode(corpus); last 5% held out; 16 evenly spaced windows of ctx tokens. Counts come from
// the training region only. Prints nats/token (what the chart's Val shows) and nats/byte (comparable
// across vocabularies and tokenizers) for: uniform, unigram, bigram (Witten-Bell backoff to unigram).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include "../mini_llm_core/tokenizer.h"

namespace
{
	constexpr size_t kMaxVocab{ 4096 };   // bigram table is vocab*vocab uint32 (64 MB at 4096)

	struct Model
	{
		size_t V;
		std::vector<uint32_t> uni;        // count of each token as a *predicted* token
		std::vector<uint32_t> ctx;        // count of each token as a context (has a successor)
		std::vector<uint32_t> types;      // distinct successors of each context
		std::vector<uint32_t> bi;         // bi[i*V + j]
		double total{};

		explicit Model(size_t v) : V{ v }, uni(v), ctx(v), types(v), bi(v * v) {}

		void train(std::vector<llm::Tokenizer::TokenId> const& ids, size_t end)
		{
			for (size_t t = 0; t < end; ++t)
				++uni[ids[t]];
			for (size_t t = 0; t + 1 < end; ++t)
			{
				auto& c{ bi[size_t(ids[t]) * V + ids[t + 1]] };
				if (!c++)
					++types[ids[t]];
				++ctx[ids[t]];
			}
			total = double(end);
		}

		double p_uni(size_t j) const { return (uni[j] + .5) / (total + .5 * double(V)); }   // add-1/2: never zero

		// Witten-Bell: lambda = c(i) / (c(i) + distinct successors of i). Unseen context -> pure unigram.
		double p_bi(size_t i, size_t j) const
		{
			double const c{ double(ctx[i]) }, T{ double(types[i]) };
			if (c == 0)
				return p_uni(j);
			double const lam{ c / (c + T) };
			return lam * double(bi[i * V + j]) / c + (1. - lam) * p_uni(j);
		}
	};
}

int main(int argc, char** argv)
{
	if (argc < 3)
	{
		std::fprintf(stderr, "usage: baseline <corpus> <vocab.bin> [ctx=64]\n");
		return 1;
	}
	size_t const CTX{ argc > 3 ? size_t(std::strtoull(argv[3], nullptr, 10)) : 64 };

	std::ifstream fc{ argv[1], std::ios::binary };
	if (!fc) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
	std::string corpus{ std::istreambuf_iterator<char>(fc), std::istreambuf_iterator<char>() };

	llm::Tokenizer tok;
	{
		std::ifstream fv{ argv[2], std::ios::binary };
		if (!fv) { std::fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
		fv >> tok;
	}
	size_t const V{ tok.vocab_size() };
	if (V > kMaxVocab) { std::fprintf(stderr, "vocab %zu > %zu: bigram table too big, raise kMaxVocab\n", V, kMaxVocab); return 1; }

	std::printf("encoding %zu bytes...\n", corpus.size());
	auto const ids{ tok.encode(corpus) };
	size_t const n{ ids.size() };
	double const bytesPerTok{ double(corpus.size()) / double(n) };
	if (n <= CTX + 1) { std::fprintf(stderr, "corpus too short\n"); return 1; }

	// Same split as CWorkerThread::Init.
	size_t const valLen{ n / 20 };
	size_t const trainEnd{ (valLen > CTX + 1 && n - valLen > CTX + 1) ? n - valLen : n };
	if (trainEnd == n) { std::fprintf(stderr, "no validation region\n"); return 1; }

	Model m{ V };
	m.train(ids, trainEnd);

	// loss of predicting ids[t+1] from ids[t], averaged like EvalVal: per window, then over windows.
	auto const nats = [&](auto&& logp_fn, size_t from, size_t count)
		{
			double s{};
			for (size_t t = from; t < from + count; ++t)
				s += logp_fn(ids[t], ids[t + 1]);
			return -s;
		};
	auto const uniform = [&](size_t, size_t) { return -std::log(double(V)); };
	auto const unigram = [&](size_t, size_t j) { return std::log(m.p_uni(j)); };
	auto const bigram = [&](size_t i, size_t j) { return std::log(m.p_bi(i, j)); };

	size_t const slots{ n - trainEnd - CTX - 1 };
	size_t const nWin{ (std::min<size_t>)(16, slots) };
	auto const windows = [&](auto&& f)
		{
			double sum{};
			for (size_t i = 0; i < nWin; ++i)
				sum += nats(f, trainEnd + i * slots / nWin, CTX) / double(CTX);
			return sum / double(nWin);
		};
	auto const whole = [&](auto&& f) { return nats(f, trainEnd, n - trainEnd - 1) / double(n - trainEnd - 1); };

	std::printf("tokens %zu  vocab %zu  bytes/token %.3f  train %zu  val %zu  ctx %zu  windows %zu\n\n",
		n, V, bytesPerTok, trainEnd, n - trainEnd, CTX, nWin);
	std::printf("%-10s %14s %14s %14s %14s\n", "model", "16-win nats/tok", "whole nats/tok", "16-win nats/B", "whole nats/B");
	auto const row = [&](char const* name, auto&& f)
		{
			double const a{ windows(f) }, b{ whole(f) };
			std::printf("%-10s %14.4f %14.4f %14.4f %14.4f\n", name, a, b, a / bytesPerTok, b / bytesPerTok);
		};
	row("uniform", uniform);
	row("unigram", unigram);
	row("bigram", bigram);
	std::printf("\n(the chart's Val is comparable to the '16-win nats/tok' column; use nats/B across vocabularies)\n");
	return 0;
}
