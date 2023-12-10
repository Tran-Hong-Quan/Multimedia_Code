#ifndef VECTOR2INT_H
#define VECTOR2INT_H

struct Vector2Int {
    int x;
    int y;

    // Constructors
    Vector2Int();
    Vector2Int(int xVal, int yVal);

    // Arithmetic operators
    Vector2Int operator+(const Vector2Int& other) const;
    Vector2Int operator-(const Vector2Int& other) const;
    Vector2Int operator*(int scalar) const;
    Vector2Int operator/(int divisor) const;

    // Comparison operators
    bool operator==(const Vector2Int& other) const;
    bool operator!=(const Vector2Int& other) const;

    // Compound assignment operators
    Vector2Int& operator+=(const Vector2Int& other);
    Vector2Int& operator-=(const Vector2Int& other);
    Vector2Int& operator*=(int scalar);
    Vector2Int& operator/=(int divisor);
};

#endif
