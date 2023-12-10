#include "Vector2Int.h"

Vector2Int::Vector2Int() : x(0), y(0) {}

Vector2Int::Vector2Int(int xVal, int yVal) : x(xVal), y(yVal) {}

Vector2Int Vector2Int::operator+(const Vector2Int& other) const {
    return Vector2Int(x + other.x, y + other.y);
}

Vector2Int Vector2Int::operator-(const Vector2Int& other) const {
    return Vector2Int(x - other.x, y - other.y);
}

Vector2Int Vector2Int::operator*(int scalar) const {
    return Vector2Int(x * scalar, y * scalar);
}

Vector2Int Vector2Int::operator/(int divisor) const {
    if (divisor != 0) {
        return Vector2Int(x / divisor, y / divisor);
    }
    else {
        return *this;
    }
}

bool Vector2Int::operator==(const Vector2Int& other) const {
    return x == other.x && y == other.y;
}

bool Vector2Int::operator!=(const Vector2Int& other) const {
    return !(*this == other);
}

Vector2Int& Vector2Int::operator+=(const Vector2Int& other) {
    x += other.x;
    y += other.y;
    return *this;
}

Vector2Int& Vector2Int::operator-=(const Vector2Int& other) {
    x -= other.x;
    y -= other.y;
    return *this;
}

Vector2Int& Vector2Int::operator*=(int scalar) {
    x *= scalar;
    y *= scalar;
    return *this;
}

Vector2Int& Vector2Int::operator/=(int divisor) {
    if (divisor != 0) {
        x /= divisor;
        y /= divisor;
    }
    else {
    }
    return *this;
}
