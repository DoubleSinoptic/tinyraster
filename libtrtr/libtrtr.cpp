// libtrtr.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//
#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <chrono>

#define STB_IMAGE_WRITE_IMPLEMENTATION 1
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION 1
#include "stb_image.h"

union COLOR_UINT_R8G8B8A8
{
    uint8_t rgba[4];

    constexpr COLOR_UINT_R8G8B8A8() : rgba{ 0, 0, 0, 0 }
    {};

    constexpr COLOR_UINT_R8G8B8A8(uint8_t r, uint8_t g, uint8_t b, uint8_t a) :
        rgba{ r, g, b, a }
    {}

    constexpr COLOR_UINT_R8G8B8A8(int r, int g, int b, int a) :
        rgba{ (uint8_t)r,  (uint8_t)g,  (uint8_t)b, (uint8_t)a }
    {}
};



template<typename C>
struct vector2 {
    C x, y;
    constexpr vector2(const vector2&) = default;
    constexpr vector2& operator=(const vector2&) = default;
    constexpr vector2(C x, C y) : x(x), y(y)
    {}
    constexpr vector2() : x(C(0)), y(C(0))
    {}
    constexpr vector2(C v) : x(v), y(v)
    {}
};
template<typename C>
constexpr vector2<C> operator-(const vector2<C>& a, const vector2<C>& b)
{
    return { a.x - b.x, a.y - b.y };
}
template<typename C>
constexpr vector2<C> operator+(const vector2<C>& a, const vector2<C>& b)
{
    return { a.x + b.x, a.y + b.y };
}
template<typename C>
constexpr vector2<C> operator*(const vector2<C>& a, const vector2<C>& b)
{
    return { a.x * b.x, a.y * b.y };
}
template<typename C>
constexpr vector2<C> min(const vector2<C>& a, const vector2<C>& b)
{
    return vector2<C> {
        a.x < b.x ? a.x : b.x,
            a.y < b.y ? a.y : b.y,
    };
}
template<typename C>
constexpr vector2<C> max(const vector2<C>& a, const vector2<C>& b)
{
    return vector2<C> {
        a.x > b.x ? a.x : b.x,
            a.y > b.y ? a.y : b.y,
    };
}
template<typename C>
constexpr vector2<C> clamp(const vector2<C>& v, const vector2<C>& b, const vector2<C>& c)
{
    return min(max(v, b), c);
}

template<typename C>
constexpr C dot(const vector2<C>& a, const vector2<C>& b)
{
    return C {
        a.x * b.x +
        a.y * b.y
    };
}

typedef vector2<float> float2;
typedef vector2<int> int2;

inline float2 floor(const float2& f)
{
    return { floorf(f.x), floorf(f.y) };
}
inline float2 fract(const float2& f)
{
    return f - floor(f);
}
union float2x2
{
    constexpr float2x2(float v) :
        col {
            {v, 0.0f},
            {0.0f, v}
        }
    {}

    constexpr float2x2() :
        col{
            {0.0f, 0.0f},
            {0.0f, 0.0f}
        }
    {}

    constexpr float2x2(float a, float b, float c, float d) :
        col{
            {a, c},
            {b, d}
    }
    {}

    float2 col[2];
    struct
    {
        float2 c0;
        float2 c1;
    };
    struct
    {
        float a, c;
        float b, d;
    };
};

constexpr float2 mul(const float2x2& m, const float2& v)
{
    return {
        dot(v, float2(m.a, m.b)),
        dot(v, float2(m.c, m.d)),
    };
}

struct alignas(16) float4
{
    float x, y, z, w;

    constexpr float4(const float4&) = default;
    constexpr float4& operator=(const float4&) = default;

    constexpr float4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w)
    {}
    constexpr float4() : x(float(0)), y(float(0)), z(float(0)), w(float(0))
    {}
    constexpr float4(float v) : x(v), y(v), z(v), w(v)
    {}
};

constexpr float4 operator-(const float4& a, const float4& b)
{
    return { a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w };
}

constexpr float4 operator*(const float4& a, const float4& b)
{
    return { a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w };
}

constexpr float4 operator+(const float4& a, const float4& b)
{
    return { a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w };
}
constexpr float4 lerp(const float4& a, const float4& b, const float4& c)
{
    return a * (float4(1.0f) - c) + b * c;
}

static constexpr const float unorm8_to_float = 1.0f / 255.0f;
static constexpr const float float_to_unorm8 = 255.0f;

constexpr COLOR_UINT_R8G8B8A8 rgbfcvt(const float4& cvt)
{
    float4 rq = cvt * float_to_unorm8;
    COLOR_UINT_R8G8B8A8 result = {};
    result.rgba[0] = rq.x;
    result.rgba[1] = rq.y;
    result.rgba[2] = rq.z;
    result.rgba[3] = rq.w;
    return result;
}

constexpr float4 frgbcvt(COLOR_UINT_R8G8B8A8 color)
{
    return {
        unorm8_to_float * color.rgba[0],
        unorm8_to_float * color.rgba[1],
        unorm8_to_float * color.rgba[2],
        unorm8_to_float * color.rgba[3]
    };
}

struct RESOURCE_REGISTER
{
    void*   memory;
    int16_t pitch;
    int2    ires;
    int2    iresminusone;
    float2  fres;
    float2  halfpixel;

    void assign(void* addr, int16_t p, int2 r)
    {
        assert(p % 4 == 0);
        pitch = p / 4;
        memory = addr;
        ires = r;
        iresminusone = r - int2(1, 1);
        fres = { (float)r.x, (float)r.y };
        halfpixel = { 0.5f / r.x, 0.5f / r.y };
    }

    inline float4 Load(const int2& c) const
    {
        const int2 index = clamp(c, int2(0, 0), iresminusone);
        const COLOR_UINT_R8G8B8A8* pixeldata = (const COLOR_UINT_R8G8B8A8*)memory;
        const auto color = pixeldata[index.x + index.y * pitch];
        return {
           unorm8_to_float* color.rgba[0],
           unorm8_to_float* color.rgba[1],
           unorm8_to_float* color.rgba[2],
           unorm8_to_float* color.rgba[3]
        };
    }

    inline float4 Sample(float2 uv) const
    {
        const float2 st = (fract(uv) - halfpixel) * fres;
        const float2 w =   fract(st);
        const float2 stf = floor(st);
        const int2   i   = int2(stf.x, stf.y);
        const float4 a =   Load(i + int2(0,0));
        const float4 b =   Load(i + int2(1,0));
        const float4 c =   Load(i + int2(0,1));
        const float4 d =   Load(i + int2(1,1));
        return lerp(lerp(a, b, w.x), lerp(c, d, w.x), w.y);
    }

};

float2x2 inverse(const float2x2& mat)
{
    float invd = 1.0f / (mat.a * mat.d - mat.b * mat.c);
    float2x2 result;
    result.c0 = { invd * mat.d, invd * -mat.c};
    result.c1 = { invd * -mat.b, invd * mat.a};
    return result;
}



struct draw 
{
    static constexpr bool edge(float xa, float ya,
        float xb, float yb,
        float xp, float yp)
    {
        return ((yp - ya) * (xb - xa) - (xp - xa) * (yb - ya)) >= 0.0f;
    }

    static constexpr const uint32_t pitch = 1024;
    static constexpr const uint32_t H = 1024;
    int2 irast = { pitch , H };
    int2 irastminusone = irast - int2(1, 1);

    COLOR_UINT_R8G8B8A8 result[pitch * H];
    COLOR_UINT_R8G8B8A8 buffer[pitch * H];

    float2 regionmin;
    float2 regionmax;

    void clear()
    {
        regionmin = {};
        regionmax = {};
        std::memset(buffer, 0, sizeof(buffer));
        std::memset(result, 0, sizeof(result));
    }

    void mosaicflush()
    {
        COLOR_UINT_R8G8B8A8* __restrict _buffer = buffer;
        COLOR_UINT_R8G8B8A8* __restrict _result = result;
        for (int y = regionmin.y; y < regionmax.y; y++)
        {
            for (int x = regionmin.x; x < regionmax.x; x++)
            {
                float4 dest = frgbcvt(_result[x + y * pitch]);
                float4 src = frgbcvt(_buffer[x + y * pitch]);

                float4 c = {};
                c.x = dest.x * (1.0 - src.w) + src.x * src.w;
                c.y = dest.y * (1.0 - src.w) + src.y * src.w;
                c.z = dest.z * (1.0 - src.w) + src.z * src.w;
                c.w = dest.w * (1.0 - src.w) + src.w;

                _result[x + y * pitch] = rgbfcvt(c);
                _buffer[x + y * pitch] = {};
            }
        }

        regionmin = {};
        regionmax = {};
    }

    template<typename Ps>
    void rast(
        float2 v0,
        float2 v1,
        float2 v2,
        float2 uv0,
        float2 uv1,
        float2 uv2,
        Ps& ps)
    {
        const float2 _max = max(max(v0, v1), v2);
        const float2 _min = min(min(v0, v1), v2);

        regionmin = min(regionmin, _min);
        regionmax = max(regionmax, _max);

        const float2 dirA = v1 - v0;
        const float2 dirB = v2 - v0;
        const float2x2 barycetnric = inverse({
            dirA.x, dirB.x,
            dirA.y, dirB.y
        });

        for (int y = _min.y; y < _max.y; y++)
        {
            for (int x = _min.x; x < _max.x; x++)
            {
                float px = x + 0.5f;
                float py = y + 0.5f;

                const float kcoff = 0.35f;

                bool insideA = 
                       edge(v0.x, v0.y, v1.x, v1.y, px + kcoff, py)
                    && edge(v1.x, v1.y, v2.x, v2.y, px + kcoff, py)
                    && edge(v2.x, v2.y, v0.x, v0.y, px + kcoff, py);

                bool insideB =
                    edge(v0.x, v0.y, v1.x, v1.y,    px - kcoff, py)
                    && edge(v1.x, v1.y, v2.x, v2.y, px - kcoff, py)
                    && edge(v2.x, v2.y, v0.x, v0.y, px - kcoff, py);

                bool insideC =
                    edge(v0.x, v0.y, v1.x, v1.y,    px, py + kcoff)
                    && edge(v1.x, v1.y, v2.x, v2.y, px, py + kcoff)
                    && edge(v2.x, v2.y, v0.x, v0.y, px, py + kcoff);

                bool insideD =
                    edge(v0.x, v0.y, v1.x, v1.y,    px, py - kcoff)
                    && edge(v1.x, v1.y, v2.x, v2.y, px, py - kcoff)
                    && edge(v2.x, v2.y, v0.x, v0.y, px, py - kcoff);

                float alpha =  
                    (insideA ? 0.25f : 0.0f) +
                    (insideB ? 0.25f : 0.0f) +
                    (insideC ? 0.25f : 0.0f) +
                    (insideD ? 0.25f : 0.0f);

                if (insideA || insideB || insideC || insideD)
                {
                    const float2 vert = float2(px, py);
                    const float2 lerpc = mul(barycetnric, vert - v0);
                    const float w = 1.0f - lerpc.x - lerpc.y;
                    const float2 uv = uv0 * float2(w) + uv1 * float2(lerpc.x) + uv2 * float2(lerpc.y);
                    const float4 color = ps.pixel(vert, uv);
                    const int2 rastcoord = clamp(int2(x, y), int2(0, 0), irastminusone);
                    const ptrdiff_t pose = rastcoord.x + pitch * rastcoord.y;
                    float balpha = buffer[pose].rgba[3] * unorm8_to_float + alpha;
                    buffer[pose] = rgbfcvt({ color.x, color.y, color.z, balpha > 1.0f ? 1.0f : balpha});
                }
            }
        }
    }
};

struct PsBliter
{
    RESOURCE_REGISTER texture;

    inline float4 pixel(const float2& vert, const float2& uv)
    {
        float pf = powf(1.0f - uv.x, 0.4f);
        return texture.Sample(uv) * float4(pf < 0.2 ? 0.2 : pf);
    }
};

struct PsBliterEnd
{
    RESOURCE_REGISTER texture;

    inline float4 pixel(const float2& vert, const float2& uv)
    {
        return texture.Sample(uv);
    }
};

draw t;

int main()
{
    

    int x, y, c;
    stbi_uc* q = stbi_load("C:\\Users\\doubl\\Desktop\\post\\A.png", &x, &y, &c, 4);

    auto ts = std::chrono::high_resolution_clock::now();

    RESOURCE_REGISTER reg;    
    reg.assign(q, x * 4, { x, y });

    PsBliter blt;
    blt.texture = reg;
    PsBliterEnd bltend;
    bltend.texture = reg;

    t.clear();

   /* for (int i = 5; i >= 0; i--)
    {
        float2 offset = float2(i * 52, 0);
        t.rast(
            float2(95, 95) + offset,
            float2(600, 15) + offset,
            float2(15, 600) + offset, float2(0, 0), float2(1, 0), float2(0, 1));

        t.rast(
            float2(15, 600) + offset,
            float2(600, 15) + offset,
            float2(600, 600) + offset, float2(0, 1), float2(1, 0), float2(1, 1));
    }
 */

    for (int i = 0; i < 4; i++)
    {
        float2 offset = float2(i * 32, 0);

        if (i == 3)
        {
            t.rast(
                float2(25, 32) + offset,
                float2(94, 20) + offset,
                float2(25, 230) + offset, float2(0, 0), float2(1, 0), float2(0, 1), bltend);

            t.rast(
                float2(25, 230) + offset,
                float2(94, 20) + offset,
                float2(94, 197) + offset, float2(0, 1), float2(1, 0), float2(1, 1), bltend);

        }
        else
        {
            t.rast(
                float2(25, 32) + offset,
                float2(94, 20) + offset,
                float2(25, 230) + offset, float2(0, 0), float2(1, 0), float2(0, 1), blt);

            t.rast(
                float2(25, 230) + offset,
                float2(94, 20) + offset,
                float2(94, 197) + offset, float2(0, 1), float2(1, 0), float2(1, 1), blt);

        }
        
        t.mosaicflush();
    }

    auto te = std::chrono::high_resolution_clock::now();

    std::cout << std::chrono::duration_cast<std::chrono::microseconds>((te - ts)).count();

    stbi_write_png("C:\\Users\\doubl\\source\\repos\\tinyraster\\libtrtr\\res.png", 1024, 1024, 4, t.result, 1024 * 4);

    return 0;
}

