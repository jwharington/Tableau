#pragma once

#include "Integration.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace tableau::integration {

enum class DOP853Status : int {
    Success = 1,
    Interrupted = 2,
    InvalidInput = -1,
    TooManySteps = -2,
    StepTooSmall = -3,
    ProbablyStiff = -4,
};

enum class DOP853OutputMode : int {
    None = 0,
    EveryAcceptedStep = 1,
    DenseEveryStep = 2,
    DenseSparse = 3,
};

struct DOP853Stats {
    int nfcn{0};
    int nstep{0};
    int naccpt{0};
    int nrejct{0};
};

struct DOP853Tolerance {
    enum class Kind {
        Scalar,
        PerComponent,
    };

    Kind kind{Kind::Scalar};
    double rtol_scalar{1e-8};
    double atol_scalar{1e-8};
    std::vector<double> rtol{};
    std::vector<double> atol{};

    static DOP853Tolerance scalar(double rtol_value, double atol_value) {
        DOP853Tolerance t;
        t.kind = Kind::Scalar;
        t.rtol_scalar = rtol_value;
        t.atol_scalar = atol_value;
        return t;
    }

    static DOP853Tolerance perComponent(
        std::vector<double> rtol_values,
        std::vector<double> atol_values) {
        DOP853Tolerance t;
        t.kind = Kind::PerComponent;
        t.rtol = std::move(rtol_values);
        t.atol = std::move(atol_values);
        return t;
    }
};

struct DOP853DenseOutput {
    std::vector<size_t> components{};
    std::vector<double> cont{};
    double xold{0.0};
    double hout{0.0};

    double evaluate(size_t component, double x) const {
        if (hout == 0.0) {
            throw std::runtime_error("DOP853DenseOutput has zero step size");
        }

        size_t idx = components.size();
        for (size_t i = 0; i < components.size(); ++i) {
            if (components[i] == component) {
                idx = i;
                break;
            }
        }

        if (idx == components.size()) {
            throw std::out_of_range("Dense output not available for requested component");
        }

        const size_t nd = components.size();
        const double s = (x - xold) / hout;
        const double s1 = 1.0 - s;

        const double conpar = cont[idx + nd * 4] +
            s * (cont[idx + nd * 5] + s1 * (cont[idx + nd * 6] + s * cont[idx + nd * 7]));

        return cont[idx] +
            s * (cont[idx + nd] +
            s1 * (cont[idx + nd * 2] +
            s * (cont[idx + nd * 3] + s1 * conpar)));
    }
};

struct DOP853StepEvent {
    int step_number{0};
    double x_old{0.0};
    double x{0.0};
    double step_size{0.0};
    double estimated_error{0.0};
    const std::vector<double>* y{nullptr};
    const DOP853DenseOutput* dense_output{nullptr};
    DOP853Stats stats{};
};

using DOP853Solout = std::function<int(const DOP853StepEvent&, double&)>;

struct DOP853RunResult {
    DOP853Status status{DOP853Status::InvalidInput};
    double x{0.0};
    std::vector<double> y{};
    double suggested_h{0.0};
    double last_error{0.0};
    DOP853Stats stats{};
};

/**
 * @brief Read-only coefficient snapshot for DOP853.
 *
 * Mapping:
 * - `a14`: {a141, a147, a148, a149, a1410, a1411, a1412, a1413}
 * - `a15`: {a151, a156, a157, a158, a1511, a1512, a1513, a1514}
 * - `a16`: {a161, a166, a167, a168, a169, a1613, a1614, a1615}
 * - `d4`: {d41, d46, d47, d48, d49, d410, d411, d412, d413, d414, d415, d416}
 * - `d5`: {d51, d56, d57, d58, d59, d510, d511, d512, d513, d514, d515, d516}
 * - `d6`: {d61, d66, d67, d68, d69, d610, d611, d612, d613, d614, d615, d616}
 * - `d7`: {d71, d76, d77, d78, d79, d710, d711, d712, d713, d714, d715, d716}
 */
struct DOP853CoefficientSet {
    std::array<double, 12> c{};
    std::array<std::array<double, 12>, 12> a{};
    std::array<double, 12> b{};
    std::array<double, 12> er{};
    double c14{0.0};
    double c15{0.0};
    double c16{0.0};
    double bhh1{0.0};
    double bhh2{0.0};
    double bhh3{0.0};
    std::array<double, 8> a14{};
    std::array<double, 8> a15{};
    std::array<double, 8> a16{};
    std::array<double, 12> d4{};
    std::array<double, 12> d5{};
    std::array<double, 12> d6{};
    std::array<double, 12> d7{};
};

template <typename State>
struct DOP853Config {
    double tolerance = 1e-8;
    double min_step = 1e-12;
    double max_step = 0.1;
    double safety_factor = 0.9;
    double absolute_tolerance = 1e-8;
    double relative_tolerance = 1e-8;
    int max_rejections = 10;
    StateValidator<State> validator{};
    StateNorm<State> norm{};

    // Advanced DOP853 controls.
    int nmax = 100000;
    int nstiff = 1000;
    double hinitial = 0.0;
    double safe = 0.9;
    double fac1 = 0.333;
    double fac2 = 6.0;
    double beta = 0.0;
    DOP853OutputMode output_mode{DOP853OutputMode::None};
    std::vector<size_t> dense_components{};
    DOP853Solout solout{};

    // Needed by full-run API to match Fortran class-style usage.
    DerivativeFunction<State> derivative{};
};

/**
 * @brief Dormand-Prince 8(5,3) integrator.
 *
 * Generic `step`/`adaptiveStep` are preserved for any State that supports
 * default construction, +, -, and scalar *.
 *
 * Full advanced integration API (`integrate`) is available when
 * `State = std::vector<double>`.
 */
template <typename State>
class DOP853Integrator : public AdaptiveIntegrator<State> {
public:
    DOP853Integrator() : DOP853Integrator(DOP853Config<State>{}) {}

    explicit DOP853Integrator(const DOP853Config<State>& config)
        : tolerance_(config.tolerance),
          min_step_(config.min_step),
          max_step_(config.max_step),
          safety_factor_(config.safety_factor),
          absolute_tolerance_(config.absolute_tolerance),
          relative_tolerance_(config.relative_tolerance),
          max_rejections_(config.max_rejections),
          validator_(config.validator),
          norm_(config.norm),
          derivative_(config.derivative),
          nmax_(config.nmax),
          nstiff_(config.nstiff),
          hinitial_(config.hinitial),
          safe_(config.safe),
          fac1_(config.fac1),
          fac2_(config.fac2),
          beta_(config.beta),
          output_mode_(config.output_mode),
          dense_components_(config.dense_components),
          solout_(config.solout) {

        if (!norm_) {
            if constexpr (std::is_arithmetic_v<State>) {
                norm_ = [](const State& s) { return std::abs(s); };
            } else if constexpr (std::is_same_v<State, std::vector<double>>) {
                norm_ = [](const State& s) {
                    double acc = 0.0;
                    for (double v : s) {
                        acc += v * v;
                    }
                    return std::sqrt(acc);
                };
            } else {
                throw std::invalid_argument("DOP853Integrator requires a norm for this state type");
            }
        }
    }

    IntegrationResult<State> step(
        const State& current_state,
        double current_time,
        double time_step,
        const DerivativeFunction<State>& derivative_func) const override {

        if (time_step <= 0.0 || time_step < min_step_ || time_step > max_step_) {
            return {};
        }
        if (validator_ && !validator_(current_state)) {
            return {};
        }

        if constexpr (std::is_same_v<State, std::vector<double>>) {
            return stepVector(current_state, current_time, time_step, derivative_func);
        } else {
            if (!computeAllStages(current_state, current_time, time_step, derivative_func)) {
                return {};
            }

            State solution = current_state + computeSolution(time_step);
            State error_vec = computeErrorEstimateVector(time_step);
            double error = estimateError(error_vec, current_state, solution);

            IntegrationResult<State> result;
            result.state = solution;
            result.time_step_used = time_step;
            result.estimated_error = error;
            result.success = true;
            result.method_used = "dop853";
            return result;
        }
    }

    IntegrationResult<State> adaptiveStep(
        const State& current_state,
        double current_time,
        double initial_time_step,
        double target_error,
        const DerivativeFunction<State>& derivative_func) const override {

        if constexpr (std::is_same_v<State, std::vector<double>>) {
            double h = initial_time_step;
            int rejections = 0;

            while (rejections <= max_rejections_) {
                if (h < min_step_ || h > max_step_) {
                    return {};
                }

                IntegrationResult<State> trial = stepVector(current_state, current_time, h, derivative_func);
                if (!trial.success) {
                    h *= 0.5;
                    ++rejections;
                    continue;
                }

                if (trial.estimated_error <= target_error || trial.estimated_error == 0.0) {
                    return trial;
                }

                h = calculateOptimalStep(h, trial.estimated_error, target_error);
                ++rejections;
            }

            return {};
        } else {
            double h = initial_time_step;
            int rejections = 0;

            while (rejections <= max_rejections_) {
                if (h < min_step_ || h > max_step_) {
                    return {};
                }

                if (!computeAllStages(current_state, current_time, h, derivative_func)) {
                    h *= 0.5;
                    ++rejections;
                    continue;
                }

                State solution = current_state + computeSolution(h);
                State error_vec = computeErrorEstimateVector(h);
                double error = estimateError(error_vec, current_state, solution);

                if (error <= target_error || error == 0.0) {
                    IntegrationResult<State> result;
                    result.state = solution;
                    result.time_step_used = h;
                    result.estimated_error = error;
                    result.success = true;
                    result.method_used = "dop853";
                    return result;
                }

                h = calculateOptimalStep(h, error, target_error);
                ++rejections;
            }

            return {};
        }
    }

    std::string getType() const override { return "dop853"; }
    int getOrder() const override { return 8; }
    static constexpr const DOP853CoefficientSet& coefficients() { return kCoefficientSet_; }

    template <typename S = State, typename std::enable_if_t<std::is_same_v<S, std::vector<double>>, int> = 0>
    DOP853RunResult integrate(
        double x0,
        const std::vector<double>& y0,
        double xend,
        const DOP853Tolerance& tol) const {

        if (!derivative_) {
            DOP853RunResult out;
            out.status = DOP853Status::InvalidInput;
            out.x = x0;
            out.y = y0;
            return out;
        }

        return integrateImpl(x0, y0, xend, tol, derivative_);
    }

    template <typename S = State, typename std::enable_if_t<std::is_same_v<S, std::vector<double>>, int> = 0>
    DOP853RunResult integrate(
        double x0,
        const std::vector<double>& y0,
        double xend,
        const DOP853Tolerance& tol,
        const DerivativeFunction<State>& derivative_func) const {

        return integrateImpl(x0, y0, xend, tol, derivative_func);
    }

    template <typename S = State, typename std::enable_if_t<std::is_same_v<S, std::vector<double>>, int> = 0>
    const DOP853DenseOutput* lastDenseOutput() const {
        return last_dense_output_ ? &(*last_dense_output_) : nullptr;
    }

private:
    static constexpr int NUM_STAGES = 12;

    static constexpr DOP853CoefficientSet buildCoefficientSet() {
        DOP853CoefficientSet set{};
        set.c = {
            0.0,
            0.526001519587677318785587544488e-01,
            0.789002279381515978178381316732e-01,
            0.118350341907227396726757197510,
            0.281649658092772603273242802490,
            0.333333333333333333333333333333,
            0.25,
            0.307692307692307692307692307692,
            0.651282051282051282051282051282,
            0.6,
            0.857142857142857142857142857142,
            1.0,
        };
        set.a = {{
            {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
            {{5.26001519587677318785587544488e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
            {{1.97250569845378994544595329183e-2, 5.91751709536136983633785987549e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
            {{2.95875854768068491816892993775e-2, 0.0, 8.87627564304205475450678981324e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
            {{2.41365134159266685502369798665e-1, 0.0, -8.84549479328286085344864962717e-1, 9.24834003261792003115737966543e-1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
            {{3.7037037037037037037037037037e-2, 0.0, 0.0, 1.70828608729473871279604482173e-1, 1.25467687566822425016691814123e-1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
            {{3.7109375e-2, 0.0, 0.0, 1.70252211019544039314978060272e-1, 6.02165389804559606850219397283e-2, -1.7578125e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
            {{3.70920001185047927108779319836e-2, 0.0, 0.0, 1.70383925712239993810214054705e-1, 1.07262030446373284651809199168e-1, -1.53194377486244017527936158236e-2, 8.27378916381402288758473766002e-3, 0.0, 0.0, 0.0, 0.0, 0.0}},
            {{6.24110958716075717114429577812e-1, 0.0, 0.0, -3.36089262944694129406857109825, -8.68219346841726006818189891453e-1, 2.75920996994467083049415600797e+1, 2.01540675504778934086186788979e+1, -4.34898841810699588477366255144e+1, 0.0, 0.0, 0.0, 0.0}},
            {{4.77662536438264365890433908527e-1, 0.0, 0.0, -2.48811461997166764192642586468, -5.90290826836842996371446475743e-1, 2.12300514481811942347288949897e+1, 1.52792336328824235832596922938e+1, -3.32882109689848629194453265587e+1, -2.03312017085086261358222928593e-2, 0.0, 0.0, 0.0}},
            {{-9.3714243008598732571704021658e-1, 0.0, 0.0, 5.18637242884406370830023853209, 1.09143734899672957818500254654, -8.14978701074692612513997267357, -1.85200656599969598641566180701e+1, 2.27394870993505042818970056734e+1, 2.49360555267965238987089396762, -3.0467644718982195003823669022, 0.0, 0.0}},
            {{2.27331014751653820792359768449, 0.0, 0.0, -1.05344954667372501984066689879e+1, -2.00087205822486249909675718444, -1.79589318631187989172765950534e+1, 2.79488845294199600508499808837e+1, -2.85899827713502369474065508674, -8.87285693353062954433549289258, 1.23605671757943030647266201528e+1, 6.43392746015763530355970484046e-1, 0.0}},
        }};
        set.b = {
            5.42937341165687622380535766363e-2,
            0.0,
            0.0,
            0.0,
            0.0,
            4.45031289275240888144113950566,
            1.89151789931450038304281599044,
            -5.8012039600105847814672114227,
            3.1116436695781989440891606237e-1,
            -1.52160949662516078556178806805e-1,
            2.01365400804030348374776537501e-1,
            4.47106157277725905176885569043e-2,
        };
        set.er = {
            0.1312004499419488073250102996e-01,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.1225156446376204440720569753e+01,
            -0.4957589496572501915214079952,
            0.1664377182454986536961530415e+01,
            -0.3503288487499736816886487290,
            0.3341791187130174790297318841,
            0.8192320648511571246570742613e-01,
            -0.2235530786388629525884427845e-01,
        };
        set.c14 = 0.1;
        set.c15 = 0.2;
        set.c16 = 0.777777777777777777777777777778;
        set.bhh1 = 0.244094488188976377952755905512;
        set.bhh2 = 0.733846688281611857341361741547;
        set.bhh3 = 0.220588235294117647058823529412e-1;
        set.a14 = {
            5.61675022830479523392909219681e-2,
            2.53500210216624811088794765333e-1,
            -2.46239037470802489917441475441e-1,
            -1.24191423263816360469010140626e-1,
            1.5329179827876569731206322685e-1,
            8.20105229563468988491666602057e-3,
            7.56789766054569976138603589584e-3,
            -8.298e-3,
        };
        set.a15 = {
            3.18346481635021405060768473261e-2,
            2.83009096723667755288322961402e-2,
            5.35419883074385676223797384372e-2,
            -5.49237485713909884646569340306e-2,
            -1.08347328697249322858509316994e-4,
            3.82571090835658412954920192323e-4,
            -3.40465008687404560802977114492e-4,
            1.41312443674632500278074618366e-1,
        };
        set.a16 = {
            -4.28896301583791923408573538692e-1,
            -4.69762141536116384314449447206,
            7.68342119606259904184240953878,
            4.06898981839711007970213554331,
            3.56727187455281109270669543021e-1,
            -1.39902416515901462129418009734e-3,
            2.9475147891527723389556272149,
            -9.15095847217987001081870187138,
        };
        set.d4 = {
            -0.84289382761090128651353491142e+01,
            0.56671495351937776962531783590,
            -0.30689499459498916912797304727e+01,
            0.23846676565120698287728149680e+01,
            0.21170345824450282767155149946e+01,
            -0.87139158377797299206789907490,
            0.22404374302607882758541771650e+01,
            0.63157877876946881815570249290,
            -0.88990336451333310820698117400e-01,
            0.18148505520854727256656404962e+02,
            -0.91946323924783554000451984436e+01,
            -0.44360363875948939664310572000e+01,
        };
        set.d5 = {
            0.10427508642579134603413151009e+02,
            0.24228349177525818288430175319e+03,
            0.16520045171727028198505394887e+03,
            -0.37454675472269020279518312152e+03,
            -0.22113666853125306036270938578e+02,
            0.77334326684722638389603898808e+01,
            -0.30674084731089398182061213626e+02,
            -0.93321305264302278729567221706e+01,
            0.15697238121770843886131091075e+02,
            -0.31139403219565177677282850411e+02,
            -0.93529243588444783865713862664e+01,
            0.35816841486394083752465898540e+02,
        };
        set.d6 = {
            0.19985053242002433820987653617e+02,
            -0.38703730874935176555105901742e+03,
            -0.18917813819516756882830838328e+03,
            0.52780815920542364900561016686e+03,
            -0.11573902539959630126141871134e+02,
            0.68812326946963000169666922661e+01,
            -0.10006050966910838403183860980e+01,
            0.77771377980534432092869265740,
            -0.27782057523535084065932004339e+01,
            -0.60196695231264120758267380846e+02,
            0.84320405506677161018159903784e+02,
            0.11992291136182789328035130030e+02,
        };
        set.d7 = {
            -0.25693933462703749003312586129e+02,
            -0.15418974869023643374053993627e+03,
            -0.23152937917604549567536039109e+03,
            0.35763911791061412378285349910e+03,
            0.93405324183624310003907691704e+02,
            -0.37458323136451633156875139351e+02,
            0.10409964950896230045147246184e+03,
            0.29840293426660503123344363579e+02,
            -0.43533456590011143754432175058e+02,
            0.96324553959188282948394950600e+02,
            -0.39177261675615439165231486172e+02,
            -0.14972683625798562581422125276e+03,
        };
        return set;
    }

    static inline constexpr DOP853CoefficientSet kCoefficientSet_ = buildCoefficientSet();

    struct ButcherTableau {
        static constexpr double C[NUM_STAGES] = {
            0.0,
            0.526001519587677318785587544488e-01,
            0.789002279381515978178381316732e-01,
            0.118350341907227396726757197510,
            0.281649658092772603273242802490,
            0.333333333333333333333333333333,
            0.25,
            0.307692307692307692307692307692,
            0.651282051282051282051282051282,
            0.6,
            0.857142857142857142857142857142,
            1.0,
        };

        static constexpr double A[NUM_STAGES][NUM_STAGES] = {
            {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {5.26001519587677318785587544488e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {1.97250569845378994544595329183e-2, 5.91751709536136983633785987549e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {2.95875854768068491816892993775e-2, 0.0, 8.87627564304205475450678981324e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {2.41365134159266685502369798665e-1, 0.0, -8.84549479328286085344864962717e-1, 9.24834003261792003115737966543e-1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {3.7037037037037037037037037037e-2, 0.0, 0.0, 1.70828608729473871279604482173e-1, 1.25467687566822425016691814123e-1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {3.7109375e-2, 0.0, 0.0, 1.70252211019544039314978060272e-1, 6.02165389804559606850219397283e-2, -1.7578125e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {3.70920001185047927108779319836e-2, 0.0, 0.0, 1.70383925712239993810214054705e-1, 1.07262030446373284651809199168e-1, -1.53194377486244017527936158236e-2, 8.27378916381402288758473766002e-3, 0.0, 0.0, 0.0, 0.0, 0.0},
            {6.24110958716075717114429577812e-1, 0.0, 0.0, -3.36089262944694129406857109825, -8.68219346841726006818189891453e-1, 2.75920996994467083049415600797e+1, 2.01540675504778934086186788979e+1, -4.34898841810699588477366255144e+1, 0.0, 0.0, 0.0, 0.0},
            {4.77662536438264365890433908527e-1, 0.0, 0.0, -2.48811461997166764192642586468, -5.90290826836842996371446475743e-1, 2.12300514481811942347288949897e+1, 1.52792336328824235832596922938e+1, -3.32882109689848629194453265587e+1, -2.03312017085086261358222928593e-2, 0.0, 0.0, 0.0},
            {-9.3714243008598732571704021658e-1, 0.0, 0.0, 5.18637242884406370830023853209, 1.09143734899672957818500254654, -8.14978701074692612513997267357, -1.85200656599969598641566180701e+1, 2.27394870993505042818970056734e+1, 2.49360555267965238987089396762, -3.0467644718982195003823669022, 0.0, 0.0},
            {2.27331014751653820792359768449, 0.0, 0.0, -1.05344954667372501984066689879e+1, -2.00087205822486249909675718444, -1.79589318631187989172765950534e+1, 2.79488845294199600508499808837e+1, -2.85899827713502369474065508674, -8.87285693353062954433549289258, 1.23605671757943030647266201528e+1, 6.43392746015763530355970484046e-1, 0.0},
        };

        static constexpr double B[NUM_STAGES] = {
            5.42937341165687622380535766363e-2,
            0.0,
            0.0,
            0.0,
            0.0,
            4.45031289275240888144113950566,
            1.89151789931450038304281599044,
            -5.8012039600105847814672114227,
            3.1116436695781989440891606237e-1,
            -1.52160949662516078556178806805e-1,
            2.01365400804030348374776537501e-1,
            4.47106157277725905176885569043e-2,
        };

        static constexpr double ER[NUM_STAGES] = {
            0.1312004499419488073250102996e-01,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.1225156446376204440720569753e+01,
            -0.4957589496572501915214079952,
            0.1664377182454986536961530415e+01,
            -0.3503288487499736816886487290,
            0.3341791187130174790297318841,
            0.8192320648511571246570742613e-01,
            -0.2235530786388629525884427845e-01,
        };
    };

    struct DenseCoefficients {
        static constexpr double C14 = 0.1;
        static constexpr double C15 = 0.2;
        static constexpr double C16 = 0.777777777777777777777777777778;

        static constexpr double BHH1 = 0.244094488188976377952755905512;
        static constexpr double BHH2 = 0.733846688281611857341361741547;
        static constexpr double BHH3 = 0.220588235294117647058823529412e-1;

        static constexpr double A141 = 5.61675022830479523392909219681e-2;
        static constexpr double A147 = 2.53500210216624811088794765333e-1;
        static constexpr double A148 = -2.46239037470802489917441475441e-1;
        static constexpr double A149 = -1.24191423263816360469010140626e-1;
        static constexpr double A1410 = 1.5329179827876569731206322685e-1;
        static constexpr double A1411 = 8.20105229563468988491666602057e-3;
        static constexpr double A1412 = 7.56789766054569976138603589584e-3;
        static constexpr double A1413 = -8.298e-3;

        static constexpr double A151 = 3.18346481635021405060768473261e-2;
        static constexpr double A156 = 2.83009096723667755288322961402e-2;
        static constexpr double A157 = 5.35419883074385676223797384372e-2;
        static constexpr double A158 = -5.49237485713909884646569340306e-2;
        static constexpr double A1511 = -1.08347328697249322858509316994e-4;
        static constexpr double A1512 = 3.82571090835658412954920192323e-4;
        static constexpr double A1513 = -3.40465008687404560802977114492e-4;
        static constexpr double A1514 = 1.41312443674632500278074618366e-1;

        static constexpr double A161 = -4.28896301583791923408573538692e-1;
        static constexpr double A166 = -4.69762141536116384314449447206;
        static constexpr double A167 = 7.68342119606259904184240953878;
        static constexpr double A168 = 4.06898981839711007970213554331;
        static constexpr double A169 = 3.56727187455281109270669543021e-1;
        static constexpr double A1613 = -1.39902416515901462129418009734e-3;
        static constexpr double A1614 = 2.9475147891527723389556272149;
        static constexpr double A1615 = -9.15095847217987001081870187138;

        static constexpr double D41 = -0.84289382761090128651353491142e+01;
        static constexpr double D46 = 0.56671495351937776962531783590;
        static constexpr double D47 = -0.30689499459498916912797304727e+01;
        static constexpr double D48 = 0.23846676565120698287728149680e+01;
        static constexpr double D49 = 0.21170345824450282767155149946e+01;
        static constexpr double D410 = -0.87139158377797299206789907490;
        static constexpr double D411 = 0.22404374302607882758541771650e+01;
        static constexpr double D412 = 0.63157877876946881815570249290;
        static constexpr double D413 = -0.88990336451333310820698117400e-01;
        static constexpr double D414 = 0.18148505520854727256656404962e+02;
        static constexpr double D415 = -0.91946323924783554000451984436e+01;
        static constexpr double D416 = -0.44360363875948939664310572000e+01;

        static constexpr double D51 = 0.10427508642579134603413151009e+02;
        static constexpr double D56 = 0.24228349177525818288430175319e+03;
        static constexpr double D57 = 0.16520045171727028198505394887e+03;
        static constexpr double D58 = -0.37454675472269020279518312152e+03;
        static constexpr double D59 = -0.22113666853125306036270938578e+02;
        static constexpr double D510 = 0.77334326684722638389603898808e+01;
        static constexpr double D511 = -0.30674084731089398182061213626e+02;
        static constexpr double D512 = -0.93321305264302278729567221706e+01;
        static constexpr double D513 = 0.15697238121770843886131091075e+02;
        static constexpr double D514 = -0.31139403219565177677282850411e+02;
        static constexpr double D515 = -0.93529243588444783865713862664e+01;
        static constexpr double D516 = 0.35816841486394083752465898540e+02;

        static constexpr double D61 = 0.19985053242002433820987653617e+02;
        static constexpr double D66 = -0.38703730874935176555105901742e+03;
        static constexpr double D67 = -0.18917813819516756882830838328e+03;
        static constexpr double D68 = 0.52780815920542364900561016686e+03;
        static constexpr double D69 = -0.11573902539959630126141871134e+02;
        static constexpr double D610 = 0.68812326946963000169666922661e+01;
        static constexpr double D611 = -0.10006050966910838403183860980e+01;
        static constexpr double D612 = 0.77771377980534432092869265740;
        static constexpr double D613 = -0.27782057523535084065932004339e+01;
        static constexpr double D614 = -0.60196695231264120758267380846e+02;
        static constexpr double D615 = 0.84320405506677161018159903784e+02;
        static constexpr double D616 = 0.11992291136182789328035130030e+02;

        static constexpr double D71 = -0.25693933462703749003312586129e+02;
        static constexpr double D76 = -0.15418974869023643374053993627e+03;
        static constexpr double D77 = -0.23152937917604549567536039109e+03;
        static constexpr double D78 = 0.35763911791061412378285349910e+03;
        static constexpr double D79 = 0.93405324183624310003907691704e+02;
        static constexpr double D710 = -0.37458323136451633156875139351e+02;
        static constexpr double D711 = 0.10409964950896230045147246184e+03;
        static constexpr double D712 = 0.29840293426660503123344363579e+02;
        static constexpr double D713 = -0.43533456590011143754432175058e+02;
        static constexpr double D714 = 0.96324553959188282948394950600e+02;
        static constexpr double D715 = -0.39177261675615439165231486172e+02;
        static constexpr double D716 = -0.14972683625798562581422125276e+03;
    };

    struct VectorTrial {
        std::vector<double> k2;
        std::vector<double> k3;
        std::vector<double> k6;
        std::vector<double> k7;
        std::vector<double> k8;
        std::vector<double> k9;
        std::vector<double> k10;
        std::vector<double> y_stage12;
        std::vector<double> k4_combined;
        std::vector<double> y_candidate;
        double error{0.0};
    };

    double tolerance_{1e-8};
    double min_step_{1e-12};
    double max_step_{0.1};
    double safety_factor_{0.9};
    double absolute_tolerance_{1e-8};
    double relative_tolerance_{1e-8};
    int max_rejections_{10};
    StateValidator<State> validator_{};
    StateNorm<State> norm_{};
    mutable std::array<State, NUM_STAGES> k_buffer_{};

    DerivativeFunction<State> derivative_{};
    int nmax_{100000};
    int nstiff_{1000};
    double hinitial_{0.0};
    double safe_{0.9};
    double fac1_{0.333};
    double fac2_{6.0};
    double beta_{0.0};
    DOP853OutputMode output_mode_{DOP853OutputMode::None};
    std::vector<size_t> dense_components_{};
    DOP853Solout solout_{};
    mutable std::optional<DOP853DenseOutput> last_dense_output_{};

    bool computeAllStages(
        const State& current_state,
        double current_time,
        double time_step_h,
        const DerivativeFunction<State>& derivative_func) const {

        k_buffer_[0] = derivative_func(current_state, current_time);
        if (validator_ && !validator_(k_buffer_[0])) {
            return false;
        }

        for (int i = 1; i < NUM_STAGES; ++i) {
            State sum{};
            for (int j = 0; j < i; ++j) {
                sum = sum + k_buffer_[j] * (ButcherTableau::A[i][j] * time_step_h);
            }

            State stage_state = current_state + sum;
            const double stage_time = current_time + ButcherTableau::C[i] * time_step_h;

            k_buffer_[i] = derivative_func(stage_state, stage_time);
            if (validator_ && !validator_(k_buffer_[i])) {
                return false;
            }
        }

        return true;
    }

    State computeSolution(double time_step_h) const {
        State solution{};
        for (int i = 0; i < NUM_STAGES; ++i) {
            solution = solution + k_buffer_[i] * (ButcherTableau::B[i] * time_step_h);
        }
        return solution;
    }

    State computeErrorEstimateVector(double time_step_h) const {
        State error{};
        for (int i = 0; i < NUM_STAGES; ++i) {
            error = error + k_buffer_[i] * (ButcherTableau::ER[i] * time_step_h);
        }
        return error;
    }

    double estimateError(
        const State& err_state,
        const State& current_state,
        const State& next_state) const {

        double scale = absolute_tolerance_ +
            relative_tolerance_ * std::max(norm_(current_state), norm_(next_state));

        if (scale <= std::numeric_limits<double>::min()) {
            scale = absolute_tolerance_;
        }

        const double err_norm = norm_(err_state);
        return err_norm / scale;
    }

    double calculateOptimalStep(double current_step, double error, double target_error) const {
        constexpr double EXPO = 1.0 / 8.0;
        double factor = safety_factor_ * std::pow(target_error / std::max(error, 1e-16), EXPO);
        factor = std::clamp(factor, 0.1, 5.0);
        double proposed = current_step * factor;
        if (proposed < min_step_) {
            proposed = min_step_;
        }
        if (proposed > max_step_) {
            proposed = max_step_;
        }
        return proposed;
    }

    static int normalizeReturnCode(int irtrn) {
        if (irtrn == 0) {
            return 1;
        }
        return irtrn;
    }

    static bool isTolValid(const DOP853Tolerance& tol, size_t n) {
        if (tol.kind == DOP853Tolerance::Kind::Scalar) {
            return tol.atol_scalar > 0.0 && tol.rtol_scalar >= 0.0;
        }

        if (tol.rtol.size() != n || tol.atol.size() != n) {
            return false;
        }

        for (size_t i = 0; i < n; ++i) {
            if (tol.atol[i] <= 0.0 || tol.rtol[i] < 0.0) {
                return false;
            }
        }

        return true;
    }

    static std::vector<size_t> makeDenseComponents(
        const std::vector<size_t>& requested,
        size_t n,
        DOP853OutputMode output_mode) {

        if (output_mode != DOP853OutputMode::DenseEveryStep &&
            output_mode != DOP853OutputMode::DenseSparse) {
            return {};
        }

        std::vector<size_t> out = requested;
        if (out.empty()) {
            out.resize(n);
            for (size_t i = 0; i < n; ++i) {
                out[i] = i;
            }
        }

        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());

        for (size_t idx : out) {
            if (idx >= n) {
                return {};
            }
        }

        return out;
    }

    static void resizeVector(std::vector<double>& v, size_t n) {
        if (v.size() != n) {
            v.assign(n, 0.0);
        }
    }

    bool computeVectorTrial(
        const std::vector<double>& y,
        double x,
        double h,
        const std::vector<double>& k1,
        const DerivativeFunction<State>& derivative_func,
        const DOP853Tolerance& tol,
        VectorTrial& trial) const {

        const size_t n = y.size();
        resizeVector(trial.k2, n);
        resizeVector(trial.k3, n);
        resizeVector(trial.k6, n);
        resizeVector(trial.k7, n);
        resizeVector(trial.k8, n);
        resizeVector(trial.k9, n);
        resizeVector(trial.k10, n);
        resizeVector(trial.y_stage12, n);
        resizeVector(trial.k4_combined, n);
        resizeVector(trial.y_candidate, n);

        std::vector<double> y1(n, 0.0);
        std::vector<double> k4_stage(n, 0.0);
        std::vector<double> k5_stage(n, 0.0);

        for (size_t i = 0; i < n; ++i) {
            y1[i] = y[i] + h * ButcherTableau::A[1][0] * k1[i];
        }
        trial.k2 = derivative_func(y1, x + ButcherTableau::C[1] * h);

        for (size_t i = 0; i < n; ++i) {
            y1[i] = y[i] + h * (ButcherTableau::A[2][0] * k1[i] + ButcherTableau::A[2][1] * trial.k2[i]);
        }
        trial.k3 = derivative_func(y1, x + ButcherTableau::C[2] * h);

        for (size_t i = 0; i < n; ++i) {
            y1[i] = y[i] + h * (ButcherTableau::A[3][0] * k1[i] + ButcherTableau::A[3][2] * trial.k3[i]);
        }
        k4_stage = derivative_func(y1, x + ButcherTableau::C[3] * h);

        for (size_t i = 0; i < n; ++i) {
            y1[i] = y[i] + h * (ButcherTableau::A[4][0] * k1[i] + ButcherTableau::A[4][2] * trial.k3[i] + ButcherTableau::A[4][3] * k4_stage[i]);
        }
        k5_stage = derivative_func(y1, x + ButcherTableau::C[4] * h);

        for (size_t i = 0; i < n; ++i) {
            y1[i] = y[i] + h * (ButcherTableau::A[5][0] * k1[i] + ButcherTableau::A[5][3] * k4_stage[i] + ButcherTableau::A[5][4] * k5_stage[i]);
        }
        trial.k6 = derivative_func(y1, x + ButcherTableau::C[5] * h);

        for (size_t i = 0; i < n; ++i) {
            y1[i] = y[i] + h * (ButcherTableau::A[6][0] * k1[i] + ButcherTableau::A[6][3] * k4_stage[i] + ButcherTableau::A[6][4] * k5_stage[i] + ButcherTableau::A[6][5] * trial.k6[i]);
        }
        trial.k7 = derivative_func(y1, x + ButcherTableau::C[6] * h);

        for (size_t i = 0; i < n; ++i) {
            y1[i] = y[i] + h * (ButcherTableau::A[7][0] * k1[i] + ButcherTableau::A[7][3] * k4_stage[i] + ButcherTableau::A[7][4] * k5_stage[i] + ButcherTableau::A[7][5] * trial.k6[i] + ButcherTableau::A[7][6] * trial.k7[i]);
        }
        trial.k8 = derivative_func(y1, x + ButcherTableau::C[7] * h);

        for (size_t i = 0; i < n; ++i) {
            y1[i] = y[i] + h * (ButcherTableau::A[8][0] * k1[i] + ButcherTableau::A[8][3] * k4_stage[i] + ButcherTableau::A[8][4] * k5_stage[i] + ButcherTableau::A[8][5] * trial.k6[i] + ButcherTableau::A[8][6] * trial.k7[i] + ButcherTableau::A[8][7] * trial.k8[i]);
        }
        trial.k9 = derivative_func(y1, x + ButcherTableau::C[8] * h);

        for (size_t i = 0; i < n; ++i) {
            y1[i] = y[i] + h * (ButcherTableau::A[9][0] * k1[i] + ButcherTableau::A[9][3] * k4_stage[i] + ButcherTableau::A[9][4] * k5_stage[i] + ButcherTableau::A[9][5] * trial.k6[i] + ButcherTableau::A[9][6] * trial.k7[i] + ButcherTableau::A[9][7] * trial.k8[i] + ButcherTableau::A[9][8] * trial.k9[i]);
        }
        trial.k10 = derivative_func(y1, x + ButcherTableau::C[9] * h);

        for (size_t i = 0; i < n; ++i) {
            y1[i] = y[i] + h * (ButcherTableau::A[10][0] * k1[i] + ButcherTableau::A[10][3] * k4_stage[i] + ButcherTableau::A[10][4] * k5_stage[i] + ButcherTableau::A[10][5] * trial.k6[i] + ButcherTableau::A[10][6] * trial.k7[i] + ButcherTableau::A[10][7] * trial.k8[i] + ButcherTableau::A[10][8] * trial.k9[i] + ButcherTableau::A[10][9] * trial.k10[i]);
        }
        trial.k2 = derivative_func(y1, x + ButcherTableau::C[10] * h);

        for (size_t i = 0; i < n; ++i) {
            y1[i] = y[i] + h * (ButcherTableau::A[11][0] * k1[i] + ButcherTableau::A[11][3] * k4_stage[i] + ButcherTableau::A[11][4] * k5_stage[i] + ButcherTableau::A[11][5] * trial.k6[i] + ButcherTableau::A[11][6] * trial.k7[i] + ButcherTableau::A[11][7] * trial.k8[i] + ButcherTableau::A[11][8] * trial.k9[i] + ButcherTableau::A[11][9] * trial.k10[i] + ButcherTableau::A[11][10] * trial.k2[i]);
        }
        trial.y_stage12 = y1;
        trial.k3 = derivative_func(y1, x + h);

        for (size_t i = 0; i < n; ++i) {
            trial.k4_combined[i] =
                ButcherTableau::B[0] * k1[i] +
                ButcherTableau::B[5] * trial.k6[i] +
                ButcherTableau::B[6] * trial.k7[i] +
                ButcherTableau::B[7] * trial.k8[i] +
                ButcherTableau::B[8] * trial.k9[i] +
                ButcherTableau::B[9] * trial.k10[i] +
                ButcherTableau::B[10] * trial.k2[i] +
                ButcherTableau::B[11] * trial.k3[i];

            trial.y_candidate[i] = y[i] + h * trial.k4_combined[i];
        }

        double err = 0.0;
        double err2 = 0.0;

        for (size_t i = 0; i < n; ++i) {
            double sk = 0.0;
            if (tol.kind == DOP853Tolerance::Kind::Scalar) {
                sk = tol.atol_scalar + tol.rtol_scalar * std::max(std::abs(y[i]), std::abs(trial.y_candidate[i]));
            } else {
                sk = tol.atol[i] + tol.rtol[i] * std::max(std::abs(y[i]), std::abs(trial.y_candidate[i]));
            }

            const double erri2 = trial.k4_combined[i] -
                DenseCoefficients::BHH1 * k1[i] -
                DenseCoefficients::BHH2 * trial.k9[i] -
                DenseCoefficients::BHH3 * trial.k3[i];
            err2 += (erri2 / sk) * (erri2 / sk);

            const double erri =
                ButcherTableau::ER[0] * k1[i] +
                ButcherTableau::ER[5] * trial.k6[i] +
                ButcherTableau::ER[6] * trial.k7[i] +
                ButcherTableau::ER[7] * trial.k8[i] +
                ButcherTableau::ER[8] * trial.k9[i] +
                ButcherTableau::ER[9] * trial.k10[i] +
                ButcherTableau::ER[10] * trial.k2[i] +
                ButcherTableau::ER[11] * trial.k3[i];
            err += (erri / sk) * (erri / sk);
        }

        double deno = err + 0.01 * err2;
        if (deno <= 0.0) {
            deno = 1.0;
        }
        trial.error = std::abs(h) * err * std::sqrt(1.0 / (static_cast<double>(n) * deno));

        if (validator_ && !validator_(trial.y_candidate)) {
            return false;
        }

        return true;
    }

    DOP853DenseOutput buildDenseOutput(
        const std::vector<size_t>& dense_components,
        const std::vector<double>& y,
        const std::vector<double>& y_candidate,
        const std::vector<double>& k1,
        const std::vector<double>& k4_endpoint,
        const VectorTrial& trial,
        double x,
        double h,
        const DerivativeFunction<State>& derivative_func,
        int& nfcn) const {

        const size_t n = y.size();
        const size_t nrd = dense_components.size();
        DOP853DenseOutput dense;
        dense.components.resize(nrd);
        for (size_t j = 0; j < nrd; ++j) {
            dense.components[j] = static_cast<size_t>(dense_components[j]);
        }
        dense.cont.assign(8 * nrd, 0.0);
        dense.xold = x;
        dense.hout = h;

        for (size_t j = 0; j < nrd; ++j) {
            const size_t i = dense_components[j];
            const double ydiff = y_candidate[i] - y[i];
            const double bspl = h * k1[i] - ydiff;

            dense.cont[j] = y[i];
            dense.cont[j + nrd] = ydiff;
            dense.cont[j + nrd * 2] = bspl;
            dense.cont[j + nrd * 3] = ydiff - h * k4_endpoint[i] - bspl;

            dense.cont[j + nrd * 4] =
                DenseCoefficients::D41 * k1[i] +
                DenseCoefficients::D46 * trial.k6[i] +
                DenseCoefficients::D47 * trial.k7[i] +
                DenseCoefficients::D48 * trial.k8[i] +
                DenseCoefficients::D49 * trial.k9[i] +
                DenseCoefficients::D410 * trial.k10[i] +
                DenseCoefficients::D411 * trial.k2[i] +
                DenseCoefficients::D412 * trial.k3[i];

            dense.cont[j + nrd * 5] =
                DenseCoefficients::D51 * k1[i] +
                DenseCoefficients::D56 * trial.k6[i] +
                DenseCoefficients::D57 * trial.k7[i] +
                DenseCoefficients::D58 * trial.k8[i] +
                DenseCoefficients::D59 * trial.k9[i] +
                DenseCoefficients::D510 * trial.k10[i] +
                DenseCoefficients::D511 * trial.k2[i] +
                DenseCoefficients::D512 * trial.k3[i];

            dense.cont[j + nrd * 6] =
                DenseCoefficients::D61 * k1[i] +
                DenseCoefficients::D66 * trial.k6[i] +
                DenseCoefficients::D67 * trial.k7[i] +
                DenseCoefficients::D68 * trial.k8[i] +
                DenseCoefficients::D69 * trial.k9[i] +
                DenseCoefficients::D610 * trial.k10[i] +
                DenseCoefficients::D611 * trial.k2[i] +
                DenseCoefficients::D612 * trial.k3[i];

            dense.cont[j + nrd * 7] =
                DenseCoefficients::D71 * k1[i] +
                DenseCoefficients::D76 * trial.k6[i] +
                DenseCoefficients::D77 * trial.k7[i] +
                DenseCoefficients::D78 * trial.k8[i] +
                DenseCoefficients::D79 * trial.k9[i] +
                DenseCoefficients::D710 * trial.k10[i] +
                DenseCoefficients::D711 * trial.k2[i] +
                DenseCoefficients::D712 * trial.k3[i];
        }

        std::vector<double> y1(n, 0.0);
        std::vector<double> k10_extra(n, 0.0);
        std::vector<double> k2_extra(n, 0.0);
        std::vector<double> k3_extra(n, 0.0);

        for (size_t i = 0; i < n; ++i) {
            y1[i] = y[i] + h * (
                DenseCoefficients::A141 * k1[i] +
                DenseCoefficients::A147 * trial.k7[i] +
                DenseCoefficients::A148 * trial.k8[i] +
                DenseCoefficients::A149 * trial.k9[i] +
                DenseCoefficients::A1410 * trial.k10[i] +
                DenseCoefficients::A1411 * trial.k2[i] +
                DenseCoefficients::A1412 * trial.k3[i] +
                DenseCoefficients::A1413 * k4_endpoint[i]);
        }
        k10_extra = derivative_func(y1, x + DenseCoefficients::C14 * h);

        for (size_t i = 0; i < n; ++i) {
            y1[i] = y[i] + h * (
                DenseCoefficients::A151 * k1[i] +
                DenseCoefficients::A156 * trial.k6[i] +
                DenseCoefficients::A157 * trial.k7[i] +
                DenseCoefficients::A158 * trial.k8[i] +
                DenseCoefficients::A1511 * trial.k2[i] +
                DenseCoefficients::A1512 * trial.k3[i] +
                DenseCoefficients::A1513 * k4_endpoint[i] +
                DenseCoefficients::A1514 * k10_extra[i]);
        }
        k2_extra = derivative_func(y1, x + DenseCoefficients::C15 * h);

        for (size_t i = 0; i < n; ++i) {
            y1[i] = y[i] + h * (
                DenseCoefficients::A161 * k1[i] +
                DenseCoefficients::A166 * trial.k6[i] +
                DenseCoefficients::A167 * trial.k7[i] +
                DenseCoefficients::A168 * trial.k8[i] +
                DenseCoefficients::A169 * trial.k9[i] +
                DenseCoefficients::A1613 * k4_endpoint[i] +
                DenseCoefficients::A1614 * k10_extra[i] +
                DenseCoefficients::A1615 * k2_extra[i]);
        }
        k3_extra = derivative_func(y1, x + DenseCoefficients::C16 * h);

        nfcn += 3;

        for (size_t j = 0; j < nrd; ++j) {
            const size_t i = dense_components[j];

            dense.cont[j + nrd * 4] = h * (dense.cont[j + nrd * 4] +
                DenseCoefficients::D413 * k4_endpoint[i] +
                DenseCoefficients::D414 * k10_extra[i] +
                DenseCoefficients::D415 * k2_extra[i] +
                DenseCoefficients::D416 * k3_extra[i]);

            dense.cont[j + nrd * 5] = h * (dense.cont[j + nrd * 5] +
                DenseCoefficients::D513 * k4_endpoint[i] +
                DenseCoefficients::D514 * k10_extra[i] +
                DenseCoefficients::D515 * k2_extra[i] +
                DenseCoefficients::D516 * k3_extra[i]);

            dense.cont[j + nrd * 6] = h * (dense.cont[j + nrd * 6] +
                DenseCoefficients::D613 * k4_endpoint[i] +
                DenseCoefficients::D614 * k10_extra[i] +
                DenseCoefficients::D615 * k2_extra[i] +
                DenseCoefficients::D616 * k3_extra[i]);

            dense.cont[j + nrd * 7] = h * (dense.cont[j + nrd * 7] +
                DenseCoefficients::D713 * k4_endpoint[i] +
                DenseCoefficients::D714 * k10_extra[i] +
                DenseCoefficients::D715 * k2_extra[i] +
                DenseCoefficients::D716 * k3_extra[i]);
        }

        return dense;
    }

    double estimateInitialStep(
        const std::vector<double>& y,
        double x,
        double posneg,
        const std::vector<double>& f0,
        int iord,
        double hmax,
        const DOP853Tolerance& tol,
        const DerivativeFunction<State>& derivative_func) const {

        const size_t n = y.size();
        double dnf = 0.0;
        double dny = 0.0;

        for (size_t i = 0; i < n; ++i) {
            double sk = 0.0;
            if (tol.kind == DOP853Tolerance::Kind::Scalar) {
                sk = tol.atol_scalar + tol.rtol_scalar * std::abs(y[i]);
            } else {
                sk = tol.atol[i] + tol.rtol[i] * std::abs(y[i]);
            }

            dnf += (f0[i] / sk) * (f0[i] / sk);
            dny += (y[i] / sk) * (y[i] / sk);
        }

        double h = 0.0;
        if (dnf <= 1.0e-10 || dny <= 1.0e-10) {
            h = 1.0e-6;
        } else {
            h = std::sqrt(dny / dnf) * 0.01;
        }

        h = std::min(h, hmax);
        h = std::copysign(h, posneg);

        std::vector<double> y1(n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            y1[i] = y[i] + h * f0[i];
        }

        std::vector<double> f1 = derivative_func(y1, x + h);

        double der2 = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double sk = 0.0;
            if (tol.kind == DOP853Tolerance::Kind::Scalar) {
                sk = tol.atol_scalar + tol.rtol_scalar * std::abs(y[i]);
            } else {
                sk = tol.atol[i] + tol.rtol[i] * std::abs(y[i]);
            }

            const double d = (f1[i] - f0[i]) / sk;
            der2 += d * d;
        }
        der2 = std::sqrt(der2) / h;

        const double der12 = std::max(std::abs(der2), std::sqrt(dnf));
        double h1 = 0.0;
        if (der12 <= 1.0e-15) {
            h1 = std::max(1.0e-6, std::abs(h) * 1.0e-3);
        } else {
            h1 = std::pow(0.01 / der12, 1.0 / static_cast<double>(iord));
        }

        h = std::min({100.0 * std::abs(h), h1, hmax});
        return std::copysign(h, posneg);
    }

    IntegrationResult<State> stepVector(
        const std::vector<double>& current_state,
        double current_time,
        double time_step,
        const DerivativeFunction<State>& derivative_func) const {

        IntegrationResult<State> result;
        if (time_step <= 0.0) {
            return result;
        }

        if (validator_ && !validator_(current_state)) {
            return result;
        }

        const DOP853Tolerance tol = DOP853Tolerance::scalar(relative_tolerance_, absolute_tolerance_);
        std::vector<double> k1 = derivative_func(current_state, current_time);

        VectorTrial trial;
        if (!computeVectorTrial(current_state, current_time, time_step, k1, derivative_func, tol, trial)) {
            return result;
        }

        result.state = trial.y_candidate;
        result.time_step_used = time_step;
        result.estimated_error = trial.error;
        result.success = true;
        result.method_used = "dop853";
        return result;
    }

    DOP853RunResult integrateImpl(
        double x0,
        const std::vector<double>& y0,
        double xend,
        const DOP853Tolerance& tol,
        const DerivativeFunction<State>& derivative_func) const {

        DOP853RunResult out;
        out.x = x0;
        out.y = y0;
        out.suggested_h = hinitial_;
        out.last_error = 0.0;

        last_dense_output_.reset();

        if (!derivative_func || y0.empty() || !isTolValid(tol, y0.size())) {
            out.status = DOP853Status::InvalidInput;
            return out;
        }

        if (nmax_ <= 0 || safe_ >= 1.0 || safe_ <= 1.0e-4 || fac1_ <= 0.0 || fac2_ <= 0.0 || beta_ < 0.0 || beta_ > 0.2) {
            out.status = DOP853Status::InvalidInput;
            return out;
        }

        const std::vector<size_t> dense_components = makeDenseComponents(dense_components_, y0.size(), output_mode_);
        if ((output_mode_ == DOP853OutputMode::DenseEveryStep || output_mode_ == DOP853OutputMode::DenseSparse) && dense_components.empty()) {
            out.status = DOP853Status::InvalidInput;
            return out;
        }

        const double uround = std::numeric_limits<double>::epsilon();

        std::vector<double> y = y0;
        std::vector<double> k1 = derivative_func(y, x0);

        double x = x0;
        double h = hinitial_;
        double hmax = (max_step_ == 0.0) ? (xend - x) : max_step_;
        hmax = std::abs(hmax);

        const int nstiff = (nstiff_ <= 0) ? (nmax_ + 10) : nstiff_;

        out.stats.nfcn += 1;

        const double posneg = std::copysign(1.0, xend - x);
        if (h == 0.0) {
            h = estimateInitialStep(y, x, posneg, k1, 8, hmax, tol, derivative_func);
        }
        out.stats.nfcn += 1;

        double facold = 1.0e-4;
        const double expo1 = 1.0 / 8.0 - beta_ * 0.2;
        const double facc1 = 1.0 / fac1_;
        const double facc2 = 1.0 / fac2_;

        bool reject = false;
        bool last = false;
        int iasti = 0;
        int nonsti = 0;
        int irtrn = 0;
        double xout = 0.0;

        if (output_mode_ != DOP853OutputMode::None && solout_) {
            irtrn = 1;
            DOP853StepEvent event;
            event.step_number = out.stats.naccpt + 1;
            event.x_old = x;
            event.x = x;
            event.step_size = 0.0;
            event.estimated_error = 0.0;
            event.y = &y;
            event.dense_output = nullptr;
            event.stats = out.stats;
            irtrn = normalizeReturnCode(solout_(event, xout));
            if (irtrn < 0) {
                out.status = DOP853Status::Interrupted;
                out.x = x;
                out.y = y;
                out.suggested_h = h;
                return out;
            }
        }

        VectorTrial trial;

        while (true) {
            if (out.stats.nstep > nmax_) {
                out.status = DOP853Status::TooManySteps;
                break;
            }

            if (0.1 * std::abs(h) <= std::abs(x) * uround) {
                out.status = DOP853Status::StepTooSmall;
                break;
            }

            if ((x + 1.01 * h - xend) * posneg > 0.0) {
                h = xend - x;
                last = true;
            }

            out.stats.nstep += 1;

            if (irtrn >= 2) {
                k1 = derivative_func(y, x);
            }

            if (!computeVectorTrial(y, x, h, k1, derivative_func, tol, trial)) {
                out.status = DOP853Status::InvalidInput;
                break;
            }

            out.stats.nfcn += 11;

            const double err = trial.error;
            out.last_error = err;

            const double fac11 = std::pow(std::max(err, 1e-16), expo1);
            double fac = fac11 / std::pow(facold, beta_);
            fac = std::max(facc2, std::min(facc1, fac / safe_));
            double hnew = h / fac;

            if (err <= 1.0) {
                facold = std::max(err, 1.0e-4);
                out.stats.naccpt += 1;

                std::vector<double> k4_endpoint = derivative_func(trial.y_candidate, x + h);
                out.stats.nfcn += 1;

                if (out.stats.naccpt % nstiff == 0 || iasti > 0) {
                    double stnum = 0.0;
                    double stden = 0.0;
                    for (size_t i = 0; i < y.size(); ++i) {
                        const double a = k4_endpoint[i] - trial.k3[i];
                        const double b = trial.y_candidate[i] - trial.y_stage12[i];
                        stnum += a * a;
                        stden += b * b;
                    }

                    if (stden > 0.0) {
                        const double hlamb = std::abs(h) * std::sqrt(stnum / stden);
                        if (hlamb > 6.1) {
                            nonsti = 0;
                            iasti += 1;
                            if (iasti == 15) {
                                out.status = DOP853Status::ProbablyStiff;
                                out.x = x;
                                out.y = y;
                                out.suggested_h = h;
                                out.stats = out.stats;
                                return out;
                            }
                        } else {
                            nonsti += 1;
                            if (nonsti == 6) {
                                iasti = 0;
                            }
                        }
                    }
                }

                bool do_dense = false;
                bool sparse_event = false;
                if (output_mode_ == DOP853OutputMode::DenseEveryStep) {
                    do_dense = true;
                } else if (output_mode_ == DOP853OutputMode::DenseSparse && solout_) {
                    sparse_event = (xout <= x + h);
                    do_dense = sparse_event;
                }

                if (do_dense) {
                    last_dense_output_ = buildDenseOutput(
                        dense_components,
                        y,
                        trial.y_candidate,
                        k1,
                        k4_endpoint,
                        trial,
                        x,
                        h,
                        derivative_func,
                        out.stats.nfcn);
                }

                k1 = k4_endpoint;
                const double x_old = x;
                y = trial.y_candidate;
                x += h;

                if (solout_ &&
                    (output_mode_ == DOP853OutputMode::EveryAcceptedStep ||
                     output_mode_ == DOP853OutputMode::DenseEveryStep ||
                     (output_mode_ == DOP853OutputMode::DenseSparse && sparse_event))) {

                    DOP853StepEvent event;
                    event.step_number = out.stats.naccpt + 1;
                    event.x_old = x_old;
                    event.x = x;
                    event.step_size = x - x_old;
                    event.estimated_error = err;
                    event.y = &y;
                    event.dense_output = last_dense_output_ ? &(*last_dense_output_) : nullptr;
                    event.stats = out.stats;

                    irtrn = normalizeReturnCode(solout_(event, xout));
                    if (irtrn < 0) {
                        out.status = DOP853Status::Interrupted;
                        out.x = x;
                        out.y = y;
                        out.suggested_h = hnew;
                        return out;
                    }
                }

                if (last) {
                    out.status = DOP853Status::Success;
                    out.x = x;
                    out.y = y;
                    out.suggested_h = hnew;
                    return out;
                }

                if (std::abs(hnew) > hmax) {
                    hnew = posneg * hmax;
                }
                if (reject) {
                    hnew = posneg * std::min(std::abs(hnew), std::abs(h));
                }
                reject = false;
                h = hnew;
            } else {
                hnew = h / std::min(facc1, fac11 / safe_);
                reject = true;
                if (out.stats.naccpt >= 1) {
                    out.stats.nrejct += 1;
                }
                last = false;
                h = hnew;
            }
        }

        out.x = x;
        out.y = y;
        out.suggested_h = h;
        return out;
    }
};

template <typename State>
constexpr double DOP853Integrator<State>::ButcherTableau::C[NUM_STAGES];

template <typename State>
constexpr double DOP853Integrator<State>::ButcherTableau::A[NUM_STAGES][NUM_STAGES];

template <typename State>
constexpr double DOP853Integrator<State>::ButcherTableau::B[NUM_STAGES];

template <typename State>
constexpr double DOP853Integrator<State>::ButcherTableau::ER[NUM_STAGES];

} // namespace tableau::integration
