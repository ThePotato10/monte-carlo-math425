// Bus (inspection) paradox: Monte Carlo simulation on real headway data.
//
// Data: "Bus Paradox Data Log" sheet, Log tab (revision 556). Each gap is the
// time between consecutive buses at the stop, recorded as minutes + seconds.
// First-bus-of-session rows (wait from arriving at the stop, not a headway)
// are already left out. Two gaps were in progress at the planned end of their
// session; the sheet's exclusion note explains why they size-bias the sample,
// so they are marked `flagged` and dropped unless --include-flagged is given.
//
// A rider arriving at a uniformly random time lands in a gap with probability
// proportional to its length, so
//     E[wait] = E[X^2] / (2 E[X]) = (mu/2) (1 + CV^2),
// not the naive mu/2. The simulations below check this against the data.
//
// Build:  g++ -std=c++17 -O2 -o monte-carlo monte-carlo.cpp
// Run:    ./monte-carlo [--trials N] [--seed S] [--include-flagged]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

struct Gap {
    const char* session;
    const char* window;  // "Morning" or "Afternoon"
    int min;
    int sec;
    bool flagged;  // in progress at the planned session end
    double minutes() const { return min + sec / 60.0; }
};

// Log rows 3-21, 23-43, 45-65, 67-85, in order of arrival.
static const Gap kGaps[] = {
    {"Mon Morning", "Morning", 5, 1, false},
    {"Mon Morning", "Morning", 5, 25, false},
    {"Mon Morning", "Morning", 4, 21, false},
    {"Mon Morning", "Morning", 6, 24, false},
    {"Mon Morning", "Morning", 3, 51, false},
    {"Mon Morning", "Morning", 8, 2, false},
    {"Mon Morning", "Morning", 12, 17, false},
    {"Mon Morning", "Morning", 4, 3, false},
    {"Mon Morning", "Morning", 3, 40, false},
    {"Mon Morning", "Morning", 6, 29, false},
    {"Mon Morning", "Morning", 4, 16, false},
    {"Mon Morning", "Morning", 6, 23, false},
    {"Mon Morning", "Morning", 5, 53, false},
    {"Mon Morning", "Morning", 15, 36, false},
    {"Mon Morning", "Morning", 1, 21, false},
    {"Mon Morning", "Morning", 8, 0, false},
    {"Mon Morning", "Morning", 8, 14, false},
    {"Mon Morning", "Morning", 5, 5, true},
    {"Mon Morning", "Morning", 3, 56, false},

    {"Mon Afternoon", "Afternoon", 7, 5, false},
    {"Mon Afternoon", "Afternoon", 4, 59, false},
    {"Mon Afternoon", "Afternoon", 3, 19, false},
    {"Mon Afternoon", "Afternoon", 5, 53, false},
    {"Mon Afternoon", "Afternoon", 7, 21, false},
    {"Mon Afternoon", "Afternoon", 8, 59, false},
    {"Mon Afternoon", "Afternoon", 1, 19, false},
    {"Mon Afternoon", "Afternoon", 9, 24, false},
    {"Mon Afternoon", "Afternoon", 4, 25, false},
    {"Mon Afternoon", "Afternoon", 4, 23, false},
    {"Mon Afternoon", "Afternoon", 4, 45, false},
    {"Mon Afternoon", "Afternoon", 4, 4, false},
    {"Mon Afternoon", "Afternoon", 10, 58, false},
    {"Mon Afternoon", "Afternoon", 7, 9, false},
    {"Mon Afternoon", "Afternoon", 1, 46, false},
    {"Mon Afternoon", "Afternoon", 13, 9, false},
    {"Mon Afternoon", "Afternoon", 2, 8, false},
    {"Mon Afternoon", "Afternoon", 5, 47, false},
    {"Mon Afternoon", "Afternoon", 5, 38, false},
    {"Mon Afternoon", "Afternoon", 9, 1, true},
    {"Mon Afternoon", "Afternoon", 0, 2, false},

    {"Tue Morning", "Morning", 4, 45, false},
    {"Tue Morning", "Morning", 4, 5, false},
    {"Tue Morning", "Morning", 4, 5, false},
    {"Tue Morning", "Morning", 7, 43, false},
    {"Tue Morning", "Morning", 1, 18, false},
    {"Tue Morning", "Morning", 6, 54, false},
    {"Tue Morning", "Morning", 4, 1, false},
    {"Tue Morning", "Morning", 4, 57, false},
    {"Tue Morning", "Morning", 8, 12, false},
    {"Tue Morning", "Morning", 1, 26, false},
    {"Tue Morning", "Morning", 6, 31, false},
    {"Tue Morning", "Morning", 5, 21, false},
    {"Tue Morning", "Morning", 8, 2, false},
    {"Tue Morning", "Morning", 0, 20, false},
    {"Tue Morning", "Morning", 6, 25, false},
    {"Tue Morning", "Morning", 5, 23, false},
    {"Tue Morning", "Morning", 7, 57, false},
    {"Tue Morning", "Morning", 0, 20, false},
    {"Tue Morning", "Morning", 6, 14, false},
    {"Tue Morning", "Morning", 10, 46, false},
    {"Tue Morning", "Morning", 3, 48, false},

    {"Wed Afternoon", "Afternoon", 2, 45, false},
    {"Wed Afternoon", "Afternoon", 12, 11, false},
    {"Wed Afternoon", "Afternoon", 2, 50, false},
    {"Wed Afternoon", "Afternoon", 5, 38, false},
    {"Wed Afternoon", "Afternoon", 6, 11, false},
    {"Wed Afternoon", "Afternoon", 5, 48, false},
    {"Wed Afternoon", "Afternoon", 3, 6, false},
    {"Wed Afternoon", "Afternoon", 2, 55, false},
    {"Wed Afternoon", "Afternoon", 3, 27, false},
    {"Wed Afternoon", "Afternoon", 9, 30, false},
    {"Wed Afternoon", "Afternoon", 5, 19, false},
    {"Wed Afternoon", "Afternoon", 9, 9, false},
    {"Wed Afternoon", "Afternoon", 8, 40, false},
    {"Wed Afternoon", "Afternoon", 12, 27, false},
    {"Wed Afternoon", "Afternoon", 6, 2, false},
    {"Wed Afternoon", "Afternoon", 8, 39, false},
    {"Wed Afternoon", "Afternoon", 3, 16, false},
    {"Wed Afternoon", "Afternoon", 5, 57, false},
    {"Wed Afternoon", "Afternoon", 5, 56, false},
};

// Sessions tab: stopwatch wait from arriving at the stop to the first bus.
static const double kMeasuredFirstBusWaits[] = {3 + 21 / 60.0, 2 + 46 / 60.0};

static const double kScheduledHeadway = 5.0;  // minutes, Summary!C12:D12

struct Stats {
    long n = 0;
    double mean = 0, m2 = 0;  // Welford
    void add(double x) {
        ++n;
        double d = x - mean;
        mean += d / n;
        m2 += d * (x - mean);
    }
    double var() const { return n > 1 ? m2 / (n - 1) : 0; }
    double sd() const { return std::sqrt(var()); }
    double se() const { return sd() / std::sqrt((double)n); }
};

struct SimResult {
    Stats wait;
    Stats gapCaught;  // length of the gap the rider arrived in
    long overScheduled = 0;  // waits longer than one scheduled headway
    std::vector<double> waits;

    double quantile(double q) {
        std::sort(waits.begin(), waits.end());
        return waits[(size_t)(q * (waits.size() - 1))];
    }
    double cdf(double w) const {
        return (double)std::count_if(waits.begin(), waits.end(), [w](double x) { return x <= w; }) /
               waits.size();
    }
};

// Theoretical size-biased mean wait for a sample of gaps.
static double sizeBiasedWait(const std::vector<double>& g) {
    double s1 = 0, s2 = 0;
    for (double x : g) {
        s1 += x;
        s2 += x * x;
    }
    return s2 / (2 * s1);
}

static double mean(const std::vector<double>& g) {
    double s = 0;
    for (double x : g) s += x;
    return s / g.size();
}

static double sd(const std::vector<double>& g) {
    double m = mean(g), s = 0;
    for (double x : g) s += (x - m) * (x - m);
    return std::sqrt(s / (g.size() - 1));
}

// Drop riders at uniform times on a timeline of consecutive gaps and measure
// each one's wait until the next bus.
static SimResult simulateTimeline(const std::vector<double>& timeline, long trials,
                                  std::mt19937_64& rng) {
    std::vector<double> ends(timeline.size());
    double t = 0;
    for (size_t i = 0; i < timeline.size(); ++i) ends[i] = (t += timeline[i]);

    std::uniform_real_distribution<double> arrive(0.0, t);
    SimResult r;
    r.waits.reserve(trials);
    for (long k = 0; k < trials; ++k) {
        double a = arrive(rng);
        size_t i = std::upper_bound(ends.begin(), ends.end(), a) - ends.begin();
        if (i == ends.size()) i = ends.size() - 1;
        double w = ends[i] - a;
        r.wait.add(w);
        r.waits.push_back(w);
        r.gapCaught.add(timeline[i]);
        if (w > kScheduledHeadway) ++r.overScheduled;
    }
    return r;
}

// Renewal model: buses run indefinitely with i.i.d. gaps drawn from the
// empirical distribution (bootstrap of the observed headways).
static SimResult simulateRenewal(const std::vector<double>& gaps, long trials,
                                 std::mt19937_64& rng, long timelineLen = 1000000) {
    std::uniform_int_distribution<size_t> pick(0, gaps.size() - 1);
    std::vector<double> timeline(timelineLen);
    for (double& x : timeline) x = gaps[pick(rng)];
    return simulateTimeline(timeline, trials, rng);
}

static void printResult(const char* label, const SimResult& r, double theory) {
    std::printf("  %-34s %6.3f ± %.3f min  (SD %6.3f)", label, r.wait.mean, 1.96 * r.wait.se(),
                r.wait.sd());
    if (!std::isnan(theory)) std::printf("   (theory %6.3f)", theory);
    std::printf("\n");
}

static void histogram(const char* title, const std::vector<long>& a, const std::vector<long>& b,
                      const char* aName, const char* bName, double binWidth) {
    long total_a = 0, total_b = 0;
    for (long x : a) total_a += x;
    for (long x : b) total_b += x;
    std::vector<double> pa(a.size()), pb(b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        pa[i] = (double)a[i] / total_a;
        pb[i] = (double)b[i] / total_b;
    }
    double pmax = std::max(*std::max_element(pa.begin(), pa.end()),
                           *std::max_element(pb.begin(), pb.end()));
    const int width = 40;
    std::printf("\n%s\n  ('#' = %s, '+' = %s)\n", title, aName, bName);
    for (size_t i = 0; i < a.size(); ++i) {
        int na = (int)std::lround(pa[i] / pmax * width);
        int nb = (int)std::lround(pb[i] / pmax * width);
        std::printf("  %4.0f-%-4.0f %5.1f%% |%s\n", i * binWidth, (i + 1) * binWidth, 100 * pa[i],
                    std::string(na, '#').c_str());
        std::printf("  %9s %5.1f%% |%s\n", "", 100 * pb[i], std::string(nb, '+').c_str());
    }
}

int main(int argc, char** argv) {
    long trials = 1000000;
    unsigned long long seed = 425;
    bool includeFlagged = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--trials") && i + 1 < argc) trials = std::atol(argv[++i]);
        else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) seed = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--include-flagged")) includeFlagged = true;
        else {
            std::fprintf(stderr, "usage: %s [--trials N] [--seed S] [--include-flagged]\n", argv[0]);
            return 1;
        }
    }
    std::mt19937_64 rng(seed);

    // ---- Data ---------------------------------------------------------------
    std::vector<double> gaps;
    std::map<std::string, std::vector<double>> byWindow, bySession;
    std::vector<std::string> sessionOrder;
    for (const Gap& g : kGaps) {
        if (g.flagged && !includeFlagged) continue;
        gaps.push_back(g.minutes());
        byWindow[g.window].push_back(g.minutes());
        if (!bySession.count(g.session)) sessionOrder.push_back(g.session);
        bySession[g.session].push_back(g.minutes());
    }

    const double mu = mean(gaps), sigma = sd(gaps), cv = sigma / mu;
    const double naive = mu / 2, biased = sizeBiasedWait(gaps);

    std::printf("Bus paradox Monte Carlo  (trials = %ld, seed = %llu)\n", trials, seed);
    std::printf("Data: %zu bus-to-bus gaps from %zu sessions%s\n\n", gaps.size(), sessionOrder.size(),
                includeFlagged ? " (end-of-session gaps included)" : " (2 end-of-session gaps excluded)");
    std::printf("Observed headways\n");
    std::printf("  mean gap mu             %6.3f min\n", mu);
    std::printf("  SD sigma                %6.3f min\n", sigma);
    std::printf("  CV sigma/mu             %6.3f\n", cv);
    std::printf("  min / max               %6.3f / %.3f min\n", *std::min_element(gaps.begin(), gaps.end()),
                *std::max_element(gaps.begin(), gaps.end()));
    std::printf("  naive wait mu/2         %6.3f min\n", naive);
    std::printf("  size-biased E[X^2]/2E[X] %5.3f min  (= naive x %.3f)\n\n", biased, biased / naive);

    // ---- 1. Renewal model with bootstrap-resampled gaps --------------------
    std::printf("1. Random rider arrival, buses with i.i.d. gaps from the data\n");
    SimResult renewal = simulateRenewal(gaps, trials, rng);
    printResult("mean wait", renewal, biased);
    std::printf("  %-34s %6.3f min   (theory %6.3f; mean gap is only %.3f)\n",
                "length of gap the rider lands in", renewal.gapCaught.mean, 2 * biased, mu);
    std::printf("  %-34s %6.3f / %.3f / %.3f min\n", "median / 90th pct / max wait", renewal.quantile(0.5),
                renewal.quantile(0.9), renewal.waits.back());
    std::printf("  %-34s %6.3f   (5-min schedule: 0)\n", "P(wait > 5 min)",
                (double)renewal.overScheduled / trials);
    std::printf("\n");

    // ---- 2. Replay of the actual recorded sequences ------------------------
    // All sessions back to back: a uniform arrival over the observed time picks
    // each session in proportion to how long it ran.
    std::printf("2. Random rider arrival during the actual recorded bus sequences\n");
    std::vector<double> replay;
    for (const std::string& s : sessionOrder)
        replay.insert(replay.end(), bySession[s].begin(), bySession[s].end());
    SimResult replayAll = simulateTimeline(replay, trials, rng);
    printResult("all sessions", replayAll, biased);
    for (const std::string& s : sessionOrder) {
        SimResult r = simulateTimeline(bySession[s], trials / 4, rng);
        printResult(("  " + s).c_str(), r, sizeBiasedWait(bySession[s]));
    }
    std::printf("\n");

    // ---- 3. Morning vs afternoon -------------------------------------------
    std::printf("3. By time of day (renewal model)\n");
    for (const char* w : {"Morning", "Afternoon"}) {
        const std::vector<double>& g = byWindow[w];
        SimResult r = simulateRenewal(g, trials, rng);
        char label[64];
        std::snprintf(label, sizeof label, "%s (n=%zu, mu/2=%.2f)", w, g.size(), mean(g) / 2);
        printResult(label, r, sizeBiasedWait(g));
    }
    std::printf("\n");

    // ---- 4. Reference models -----------------------------------------------
    std::printf("4. Reference models\n");
    {
        // Buses exactly on the 5-minute schedule.
        std::vector<double> fixed(1000, kScheduledHeadway);
        printResult("perfect 5-min schedule", simulateTimeline(fixed, trials, rng), kScheduledHeadway / 2);
        // Perfectly regular at the observed mean.
        std::vector<double> regular(1000, mu);
        printResult("regular at observed mean", simulateTimeline(regular, trials, rng), mu / 2);
        // Poisson arrivals with the observed mean (CV = 1, memoryless).
        std::exponential_distribution<double> expo(1.0 / mu);
        std::vector<double> pois(1000000);
        for (double& x : pois) x = expo(rng);
        printResult("Poisson, same mean (CV=1)", simulateTimeline(pois, trials, rng), mu);
    }
    std::printf("\n");

    // ---- 5. Bootstrap confidence interval ----------------------------------
    // Sampling uncertainty from having only n gaps, which dominates the
    // simulation error above.

    // ---- Histogram ---------------------------------------------------------
    {
        const double bw = 2.0;
        const int bins = 8;
        std::vector<long> observed(bins, 0), caught(bins, 0);
        for (double x : gaps) observed[std::min(bins - 1, (int)(x / bw))]++;
        // Length-biased: each gap weighted by its length, in whole seconds.
        for (double x : gaps) caught[std::min(bins - 1, (int)(x / bw))] += std::lround(x * 60);
        histogram("Gap lengths: as recorded vs as experienced by a random rider (minutes)", observed,
                  caught, "share of gaps", "share of riders", bw);
    }
    return 0;
}
