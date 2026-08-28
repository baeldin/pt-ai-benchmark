// ---------------------------------------------------------------------------
//  A small, dependency-free path tracer with explicit next-event estimation.
//
//  Scene:   a specularly-reflective sphere (radius 1) resting on the floor of a
//           diffuse box (half-extent 2, i.e. "radius 2").  The box face at +Z is
//           left out, so the camera looks straight into the open front.  A
//           2x2 quad light sits just under the ceiling pointing straight down.
//           Everything except the sphere and the emitter is matte, so the image
//           is dominated by direct light, the sphere's mirror reflections and
//           diffuse colour bleeding.
//
//  Integrator: classic unidirectional path tracing.  Each vertex does exactly
//           one of these:
//             diffuse  -> (a) NEE: one area sample of the quad light + shadow
//                             ray, which supplies that vertex's whole direct
//                             term deterministically, plus
//                         (b) one cosine-weighted BSDF sample that carries the
//                             path on and thus supplies indirect light.
//             mirror   -> one reflected ray (NEE is pointless on a Dirac BRDF:
//                             an area-light sample always misses it).
//           A path stops when it leaves the scene (through the open front, so a
//           mirror facing that way reflects true black), hits the emitter, or is
//           killed by Russian roulette.
//
//  Output:  512x512, 8-bit sRGB, binary PPM (P6) -> "output.ppm".
//
//  Usage:   pt [spp] [out.ppm]
//
//  Build:   cl /O2 /EHsc /std:c++17 /MT main.cpp        (MSVC, or build.bat)
//          g++ -O3 -std=c++17 -pthread main.cpp -o pt   (gcc/clang)
// ---------------------------------------------------------------------------

#if defined(_MSC_VER)
#define _CRT_SECURE_NO_WARNINGS 1        // fopen/fprintf are fine for this tool
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

// ------------------------------ tunables ----------------------------------

static const int   kImageW      = 512;              // output width  (square)
static const int   kImageH      = 512;              // output height
static const int   kSpp         = 256;              // path samples per pixel
static const int   kMaxDepth    = 8;                // path vertices per sample
static const float kFovDeg      = 52.0f;            // vertical fov; the whole frustum still
                                                    // passes through the open front face
static const float kExposure    = 1.0f;             // linear exposure
static const float kLightHalf   = 1.0f;             // light quad half-extent
static const float kLightRad    = 10.0f;            // light radiance (arbitrary units)
static const float kLightY      = 1.98f;            // light height (just under the ceiling)
static const float kLightZ      = 0.35f;            // light a little forward of the sphere, so
                                                    // its reflection sits mid-sphere and the
                                                    // sphere's shadow falls on the back wall
static const float kSphereY     = -1.0f;            // sphere centre: resting on the floor
static const float kSphereZ     = -0.85f;           // ... and a bit back from centre
static const float kEps         = 1e-3f;            // shading / intersection epsilon
static const float kPi          = 3.14159265358979f;
static const char* kOutPath     = "output.ppm";     // default output (see argv)

// ------------------------------- math -------------------------------------

struct Vec3 {
    float x = 0, y = 0, z = 0;
    Vec3() = default;
    Vec3(float v) : x(v), y(v), z(v) {}
    Vec3(float X, float Y, float Z) : x(X), y(Y), z(Z) {}
};
struct Vec2 { float x = 0, y = 0; Vec2() = default; Vec2(float X, float Y) : x(X), y(Y) {} };

static inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
static inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static inline Vec3 operator*(Vec3 a, float t) { return {a.x * t, a.y * t, a.z * t}; }
static inline Vec3 operator/(Vec3 a, float t) { return {a.x / t, a.y / t, a.z / t}; }
static inline Vec3& operator+=(Vec3& a, Vec3 b) { a = a + b; return a; }
static inline Vec3& operator*=(Vec3& a, float t) { a = a * t; return a; }
// Component-wise (spectral) products: used for reflectance x radiance transport.
static inline Vec3 operator*(Vec3 a, Vec3 b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }
static inline Vec3& operator*=(Vec3& a, Vec3 b) { a = a * b; return a; }
static inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
static inline float length(Vec3 a) { return std::sqrt(dot(a, a)); }
static inline Vec3 normalize(Vec3 a) { return a / length(a); }
static inline Vec3 cmin(Vec3 a, Vec3 b) { return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)}; }
static inline Vec3 clamp01(Vec3 a) { return cmin(a, Vec3(1.0f)); }   // clamps to [0,1]

// Random numbers.  Every path gets its own xorshift32 stream, seeded from a
// splitmix64 hash of its (pixel, sample) index: no shared state, so rendering is
// thread-safe and bit-reproducible no matter how many threads are used.
static inline uint32_t seedHash(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return (uint32_t)(x ^ (x >> 31));
}

struct Rng {
    uint32_t s;
    explicit Rng(uint32_t seed) : s(seed ? seed : 1u) {}
    float next() {                                 // uniform in [0, 1)
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        return (float)(s >> 8) * (1.0f / 16777216.0f);
    }
    Vec2 next2() { return {next(), next()}; }
};

// Orthonormal tangent frame (n, u, v) around a shading normal.
struct OnBasis {
    Vec3 n, u, v;
    explicit OnBasis(Vec3 N) : n(normalize(N)) {
        // Cross n with the coordinate axis that is furthest from being parallel
        // to it: the result is guaranteed non-degenerate *and* orthogonal to n
        // (u is unit because the two inputs are orthogonal unit vectors, and so
        // is v = n x u for the same reason).
        Vec3 helper = std::fabs(n.x) > 0.9f ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
        u = normalize(cross(n, helper));
        v = cross(n, u);
    }
    // Cosine-weighted hemisphere direction (pdf == cos / pi), the Malley trick:
    // take a uniform point on the disc of radius sqrt(e2) and project it up onto
    // the hemisphere, giving the cos / pi distribution.  The result is unit
    // length already, since r^2 + h^2 == e2 + (1 - e2) == 1.
    Vec3 cosSample(float e1, float e2) const {
        float phi = 2.0f * kPi * e1;
        float r   = std::sqrt(e2);
        float h   = std::sqrt(std::max(0.0f, 1.0f - e2));
        Vec3 t1 = u * std::cos(phi) + v * std::sin(phi);   // unit, in the tangent plane
        return t1 * r + n * h;
    }
};

// ------------------------------- geometry ---------------------------------

struct Ray {
    Vec3 o, d;
    Vec3 at(float t) const { return o + d * t; }
};

enum class Mat { Diffuse, Mirror, Emissive };

// An object is either a planar quad or a sphere - all this scene needs.  A
// quad's normal is authored to face the side that gets shaded (inwards for the
// box walls, downwards for the light); intersections are still two-sided.
struct Object {
    bool   isSphere = false;
    Mat    mat      = Mat::Diffuse;
    Vec3   albedo;                       // diffuse reflectance / mirror tint
    Vec3   emission;                     // radiance, only for Mat::Emissive
    // quad: center p, unit normal n, unit in-plane axes u/v, half-extents hu/hv
    Vec3   p, n, u, v;
    float  hu = 0, hv = 0;
    // sphere: center c, radius r
    Vec3   c;
    float  r = 1;

    // Quad intersection: infinite-plane test, then bounds test in quad space.
    bool hitQuad(const Ray& ray, float tMax, float& tOut) const {
        float denom = dot(n, ray.d);
        if (std::fabs(denom) < 1e-8f) return false;                 // parallel
        float t = dot(p - ray.o, n) / denom;
        if (t < kEps || t > tMax) return false;                     // behind / too far
        Vec3 q = ray.at(t) - p;
        if (std::fabs(dot(q, u)) > hu || std::fabs(dot(q, v)) > hv) return false;
        tOut = t;
        return true;
    }

    // Sphere intersection: standard quadratic, keep the near root.
    bool hitSphere(const Ray& ray, float tMax, float& tOut) const {
        Vec3 oc = ray.o - c;
        float b = dot(oc, ray.d);
        float cc = dot(oc, oc) - r * r;
        float disc = b * b - cc;
        if (disc < 0.0f) return false;
        float t = -b - std::sqrt(disc);
        if (t < kEps || t > tMax) return false;
        tOut = t;
        return true;
    }

    bool hit(const Ray& ray, float tMax, float& tOut) const {
        return isSphere ? hitSphere(ray, tMax, tOut) : hitQuad(ray, tMax, tOut);
    }
};

struct Hit {
    float t = 0;
    Vec3  p, n;
    Mat   mat = Mat::Diffuse;
    Vec3  albedo, emission;
};

// ------------------------------- scene ------------------------------------

static std::vector<Object> g_objs;   // all occluders, with the emitter appended last
static int                 g_lightIdx = -1;   // index of the quad light in g_objs

// Append a quad surface: centre p, unit normal n, unit in-plane axes u/v,
// half-extents hu/hv, plus material.  Quads are hit from both sides (normals get
// flipped to face the ray), which is what lets the five walls also act as a
// closed shell when seen from outside.
static void addQuad(Vec3 p, Vec3 n, Vec3 u, Vec3 v, float hu, float hv,
                    Vec3 albedo, Mat mat = Mat::Diffuse, Vec3 emission = Vec3(0.0f)) {
    Object o;
    o.isSphere = false; o.mat = mat; o.albedo = albedo; o.emission = emission;
    o.p = p; o.n = n; o.u = u; o.v = v; o.hu = hu; o.hv = hv;
    g_objs.push_back(o);
}

// Assemble the scene: five walls of a box with half-extent 2 centred on the
// origin (the +Z face is left out - that is the "open front" the camera looks
// through), the mirror sphere, and the ceiling quad light.
static void buildScene() {
    const float k = 2.0f;   // box half-extent ("radius"); also each wall's half-size

    // Five inward-facing walls (normals point into the cavity).  Albedos are
    // deliberately tinted so colour bleeding between bounces - and the mirror
    // reflections of those walls - are visible in the image.
    addQuad({0, -k, 0}, {0, 1, 0}, {1, 0, 0}, {0, 0, 1}, k, k, {0.72f, 0.68f, 0.62f}); // floor
    addQuad({0,  k, 0}, {0, -1, 0}, {1, 0, 0}, {0, 0, 1}, k, k, {0.80f, 0.80f, 0.80f}); // ceiling
    addQuad({-k, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, k, k, {0.66f, 0.34f, 0.28f}); // left, warm
    addQuad({ k, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, 0, 1}, k, k, {0.28f, 0.40f, 0.66f}); // right, cool
    addQuad({0, 0, -k}, {0, 0, 1}, {1, 0, 0}, {0, 1, 0}, k, k, {0.55f, 0.55f, 0.55f}); // back

    // Specular sphere of radius 1.  Reflectance is slightly tinted and < 1 so a
    // mirror chain loses energy the way a real coating does.
    Object sph;
    sph.isSphere = true; sph.mat = Mat::Mirror;
    sph.albedo = {0.94f, 0.96f, 0.97f};
    sph.c = {0, kSphereY, kSphereZ}; sph.r = 1.0f;
    g_objs.push_back(sph);

    // Ceiling quad light, a hair below the ceiling plane, pointing straight down.
    g_lightIdx = (int)g_objs.size();
    addQuad({0, kLightY, kLightZ}, {0, -1, 0}, {1, 0, 0}, {0, 0, 1},
            kLightHalf, kLightHalf, Vec3(0.0f), Mat::Emissive,
            Vec3(1.0f, 0.96f, 0.90f) * kLightRad);
    // (shadow rays skip index g_lightIdx, primary rays do not)
}

// Closest hit over the whole scene, linearly searched (the scene is tiny).
// `skipLight` is used by shadow rays, which must not count the sampled light as
// an occluder.  Pass a null `hit` for shadow/visibility queries.
static bool intersectScene(const Ray& ray, float tMax, bool skipLight, Hit* hit) {
    bool any = false;
    for (int i = 0; i < (int)g_objs.size(); ++i) {
        if (skipLight && i == g_lightIdx) continue;
        float t;
        if (!g_objs[i].hit(ray, tMax, t)) continue;
        tMax = t;               // tighten the range: only closer hits matter now
        any = true;
        if (hit) {
            const Object& o = g_objs[i];
            Vec3 p = ray.at(t);
            Vec3 n = o.isSphere ? normalize(p - o.c) : o.n;
            // Normals are authored to face the shaded side; flip only if a ray
            // somehow arrives from behind, so shading never sees a back normal.
            if (dot(ray.d, n) > 0.0f) n = n * -1.0f;
            *hit = Hit{t, p, n, o.mat, o.albedo, o.emission};
        }
    }
    return any;
}

// Hard shadow test: is anything between `p` (exclusive) and `p + d * tMax`?
static bool occluded(Vec3 p, Vec3 d, float tMax) {
    Ray ray{p + d * kEps, d};
    return intersectScene(ray, tMax, /*skipLight=*/true, nullptr);
}

// --------------------------- sampling / shading ---------------------------

// Next-event estimation (the direct-lighting term): take one uniform area
// sample q on the quad light and evaluate the rendering equation at it,
//
//   L = Le * f * cos_s * cos_l / (pdf_area * d^2),   f = albedo / pi,
//   pdf_area = 1 / lightArea
//
// A single sample is already an unbiased estimate of the *whole* direct term at
// that vertex, because we integrate over the light (not over the BRDF): nothing
// here cancels against the path's incoming pdf, which is why the caller simply
// adds this on top of the multiplied path throughput.  The one shadow ray decides
// visibility, so the lamp is never seen through the sphere, and because the shadow
// ray targets a random lamp point the resulting shadows have area-light
// penumbrae.
static Vec3 sampleLight(const Hit& h, Rng& rng) {
    const Object& L = g_objs[g_lightIdx];

    // Uniform point on the light quad.
    Vec3 q = L.p + L.u * ((rng.next() * 2.0f - 1.0f) * L.hu)
                     + L.v * ((rng.next() * 2.0f - 1.0f) * L.hv);
    float pdfArea = 1.0f / (4.0f * L.hu * L.hv);

    Vec3 wi = q - h.p;
    float d2 = dot(wi, wi);
    if (d2 < 1e-8f) return Vec3(0.0f);
    wi = wi * (1.0f / std::sqrt(d2));

    float cosS = dot(h.n, wi);            // shading surface orientation
    float cosL = dot(L.n, wi * -1.0f);    // light faces straight down
    if (cosS <= 0.0f || cosL <= 0.0f) return Vec3(0.0f);

    if (occluded(h.p, wi, std::sqrt(d2) - kEps)) return Vec3(0.0f);   // in shadow

    Vec3 f = h.albedo * (1.0f / kPi);                    // diffuse BRDF
    return L.emission * f * (cosS * cosL / (pdfArea * d2));
}

// Filmic response (Narkowicz/ACES fit) so the blown-out light and its mirror
// highlight roll off instead of clipping hard.
static Vec3 tonemap(Vec3 c) {
    auto curve = [](float x) {
        return (x * (2.51f * x + 0.03f)) / (x * (0.24f * x + 0.14f) + 1.0f);
    };
    return {curve(c.x), curve(c.y), curve(c.z)};
}

// sRGB OETF applied to a linear (post-tonemap) value.
static Vec3 toSRGB(Vec3 c) {
    auto f = [](float u) {
        u = std::min(std::max(u, 0.0f), 1.0f);
        return u <= 0.0031308f ? 12.92f * u : 1.055f * std::pow(u, 1.0f / 2.4f) - 0.055f;
    };
    return {f(c.x), f(c.y), f(c.z)};
}

// Trace one path and return its radiance estimate.
//
// Bookkeeping rule that keeps the hybrid free of double counting: a hit on the
// emitter is only added when the path reached it through *specular* vertices
// (deltaChain), or on a primary ray.  Every direct-lighting event is therefore
// accounted for exactly once:
//   * rough vertex -> by its own sampleLight() call (NEE), never by a lucky
//     random bounce into the light, which is why deltaChain is cleared there;
//   * delta vertex -> by the reflected ray actually reaching the light, which is
//     the only mechanism available to it (and what makes the sphere show the
//     lamp).
// Since NEE integrates the direct term in closed form per vertex, and BSDF
// sampling handles everything else, no multiple-importance-sampling weight is
// needed (Veach's balance heuristic would only trade variance between the two).
static Vec3 trace(const Ray& cameraRay, Rng& rng) {
    Vec3 L(0.0f), throughput(1.0f);
    Ray ray = cameraRay;
    bool deltaChain = true;      // path has not touched a rough vertex yet

    for (int depth = 0; depth < kMaxDepth; ++depth) {
        Hit h;
        if (!intersectScene(ray, 1e30f, /*skipLight=*/false, &h)) return L;  // open front: void

        if (h.mat == Mat::Emissive) {
            if (depth == 0 || deltaChain) L += throughput * h.emission;
            return L;                       // emitters are path ends
        }

        // Direct lighting: next-event estimation on rough (non-delta) vertices.
        if (h.mat == Mat::Diffuse) {
            L += throughput * sampleLight(h, rng);
            deltaChain = false;             // from here on, NEE owns the lamp term
        }

        // One BSDF sample to continue the path.
        Vec2 e = rng.next2();
        if (h.mat == Mat::Mirror) {
            Vec3 d = ray.d - h.n * (2.0f * dot(ray.d, h.n));
            ray = Ray{h.p + d * kEps, normalize(d)};
            throughput *= h.albedo;                 // f = kr * delta, pdf cancels
        } else {
            OnBasis basis(h.n);
            Vec3 d = basis.cosSample(e.x, e.y);
            ray = Ray{h.p + d * kEps, d};
            throughput *= h.albedo;                 // (albedo/pi) * cos / (cos/pi)
        }

        // Russian roulette: kill low-contribution paths after a few bounces.
        if (depth >= 3) {
            Vec3 q = clamp01(throughput);
            float p = std::max(q.x, std::max(q.y, q.z));
            if (rng.next() > p) return L;
            throughput *= 1.0f / std::max(p, 1e-3f);
        }
    }
    return L;   // path terminated by depth limit
}

// ------------------------------- camera -----------------------------------

struct Camera {
    Vec3 pos, fwd, right, up;
    float halfH;   // sensor half-height at unit focal length

    static Camera make() {
        Camera c;
        c.pos  = {0.0f, 0.40f, 4.5f};              // in front of the open face
        c.fwd  = normalize(Vec3(0.0f, -0.01f, -1.0f)); // a hair down
        c.right = normalize(cross(c.fwd, Vec3(0, 1, 0)));
        c.up    = normalize(cross(c.right, c.fwd));
        c.halfH = std::tan(0.5f * kFovDeg * (kPi / 180.0f));
        return c;
    }

    // Primary ray for sample point (u,v) in [0,1]^2 of the image.
    Ray sample(float u, float v) const {
        // Square image: equal horizontal and vertical sensor extents.
        float sx = (2.0f * u - 1.0f) * halfH;
        float sy = (1.0f - 2.0f * v) * halfH;
        Vec3 d = normalize(fwd + right * sx + up * sy);
        return Ray{pos, d};
    }
};

// ------------------------------- render -----------------------------------

int main(int argc, char** argv) {
    // Optional overrides: pt [spp] [out.ppm]
    int         spp     = argc > 1 ? std::atoi(argv[1]) : kSpp;
    const char* outPath = argc > 2 ? argv[2] : kOutPath;
    if (spp <= 0) spp = kSpp;

    buildScene();
    Camera cam = Camera::make();

    std::vector<uint8_t> rgb((size_t)kImageW * kImageH * 3, 0);
    std::atomic<int> nextRow{0};

    const auto t0 = std::chrono::steady_clock::now();

    // Row-level parallel rendering.  Rows are independent (every path carries its
    // own seeded RNG), so the output is deterministic for any thread count.
    auto worker = [&]() {
        int y;
        while ((y = nextRow.fetch_add(1)) < kImageH) {
            for (int x = 0; x < kImageW; ++x) {
                Vec3 acc(0.0f);
                for (int s = 0; s < spp; ++s) {
                    Rng rng(seedHash(((uint64_t)y * kImageW + x) * 65536ull + (uint64_t)s));
                    // Jitter inside the pixel -> antialiased edges.
                    Vec2 j = rng.next2();
                    float u = ((float)x + j.x) / (float)kImageW;
                    float v = ((float)y + j.y) / (float)kImageH;
                    acc += trace(cam.sample(u, v), rng);
                }
                Vec3 c = acc * (1.0f / (float)spp) * kExposure;
                c = toSRGB(clamp01(tonemap(c)));      // tonemap -> sRGB -> 8 bit
                uint8_t* dst = &rgb[((size_t)y * kImageW + x) * 3];
                dst[0] = (uint8_t)std::min(255.0f, c.x * 255.0f + 0.5f);
                dst[1] = (uint8_t)std::min(255.0f, c.y * 255.0f + 0.5f);
                dst[2] = (uint8_t)std::min(255.0f, c.z * 255.0f + 0.5f);
            }
        }
    };

    unsigned nThreads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> pool;
    for (unsigned i = 0; i < nThreads; ++i) pool.emplace_back(worker);
    for (auto& t : pool) t.join();

    const double secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    // Binary PPM (P6): 8-bit sRGB-encoded channels, no alpha, no compression.
    FILE* f = std::fopen(outPath, "wb");
    if (!f) { std::fprintf(stderr, "failed to open %s\n", outPath); return 1; }
    std::fprintf(f, "P6\n%d %d\n255\n", kImageW, kImageH);
    std::fwrite(rgb.data(), 1, rgb.size(), f);
    std::fclose(f);

    std::printf("wrote %s  %dx%d  %d spp  max depth %d  in %.2fs (%u threads)\n",
                outPath, kImageW, kImageH, spp, kMaxDepth, secs, nThreads);
    return 0;
}
