import math
import numpy as np

class Spring:
    def __init__(self, k=1):
        self._k = k
    
    def get_k(self):
        return self._k
    
    def set_k(self, k):
        self._k = k
    
 
    def move(self, t, dt, x0=0, v0=0, t0=None, t1=None, m=1):
        if t0 is not None and t1 is not None:
            n = math.ceil((t1 - t0)/dt)
        else:
            n = math.ceil(t/dt)
        
        x = [x0]
        v = [v0]
        for i in range(n):
            if t0 is not None and t1 is not None:
                a = -self._k * x[-1] / m
            else:
                a = -self._k * x[-1]
            x.append(x[-1] + v[-1] * dt)
            v.append(v[-1] + a * dt)
        return x
   
 #Task 2.1   
    def inParallel(self, that):
        k_eq = self._k + that.get_k()
        return Spring(k_eq)

    def inSeries(self, that):
        k_eq = (self._k * that.get_k()) / (self._k + that.get_k())
        return Spring(k_eq)

    
#Test
spring = Spring(10)

# Using move(t, dt, x0, v0)
coordinates1 = spring.move(10, 7, 1, 2)
print(coordinates1)


# Using move(t, dt, x0, v0)
coordinates1 = spring.move(0, 7, 1, 2)
print(coordinates1)


# Using move(t, dt, x0)
coordinates2 = spring.move(5, 0.1, 2)
print(coordinates2)

# Using move(t0, t1, dt, x0, v0)
coordinates3 = spring.move(0, 5, 0.1, 2, 0)
print(coordinates3)

# Using move(t0, t1, dt, x0)
coordinates4 = spring.move(0, 5, 0.1, 2)
print(coordinates4)

# Using move(t0, t1, dt, x0, v0, m)
coordinates5 = spring.move(0, 5, 0.1, 2, 0, 1)
print(coordinates5)



    
from typing import List
    
class SpringArray(Spring):
    @staticmethod
    def equivalentSpring(spring_expr: str, springs: [Spring]) -> Spring:
        stack = []
        top = -1

        for char in spring_expr:
            if char.isdigit():
                top += 1
                stack.append(springs[int(char)])

            elif char in {'{', '['}:
                top += 1
                stack.append(Spring())

            elif char in {'}', ']'}:
                spring2, spring1 = stack[top], stack[top-1]
                top -= 2

                new_spring = spring1.inSeries(spring2) if char == '}' else spring1.inParallel(spring2)
                top += 1
                stack[top] = new_spring

        return stack[0] if top == 0 else Spring()



#Test
# Example 1
springExpr1 = "{{}}[{}]"
spring1 = SpringArray.equivalentSpring(springExpr1, None)

# Example 2
springExpr2 = "{}"
spring2 = SpringArray.equivalentSpring(springExpr1, [Spring(4), Spring(3)])
print("stiffness is ")
print(spring2.get_k())


#Task3
class FT:
    def __init__(self):
        pass

    def transform(self, data):
        n = len(data)
        amplitudes = []
        for k in range(n):
            re = 0.0
            im = 0.0
            for t in range(n):
                angle = 2 * math.pi * t * k / n
                re += data[t] * math.cos(angle)
                im -= data[t] * math.sin(angle)
            amplitude = math.sqrt(re ** 2 + im ** 2) / n
            amplitudes.append(amplitude)
        return amplitudes
values = [0, 1, 0, -1]

# Create an instance of the FT class
ft = FT()

# Calculate the Fourier transform of the sample array
amplitudes = ft.transform(values)

# Print the resulting amplitudes
print(amplitudes)


#Task4
from abc import ABC, abstractmethod
from typing import List

class Converter(ABC):
    @abstractmethod
    def bin2Dec(self, binary):
        pass

    @abstractmethod
    def bits2springs(self, bits):
        pass

    def computeOscillations(self, bits, t, dt, x0, vo):
        springs = self.bits_to_springs(bits)
        spring = springs[0]
        for i in range(1, len(springs)):
            spring = spring.inSeries(springs[i])
        body = Spring(1.0, spring)
        return body.move(t, dt, x0, vo)


    def calculate_frequency_amplitudes(self, bits):
        springs = self.convert_bits_to_springs(bits)
        oscillations = self.connect_body_to_springs(springs)
        n = len(oscillations)
        dt = 0.01
        freqs = np.fft.fftfreq(n, dt)[:n // 2]
        fft = np.fft.fft(oscillations)[:n // 2]
        amplitudes = 2.0 / n * np.abs(fft)
        return freqs, amplitudes

    def evaluate_decimal_value(self, bits):
        freqs, amplitudes = self.calculate_frequency_amplitudes(bits)
        decimal_value = 0
        for i in range(len(amplitudes)):
            decimal_value += amplitudes[i] * freqs[i]
        return decimal_value

#Task5
class Converter8Bit(Converter):
    def bin2Dec(binary):
        decimal = 0
        for i, bit in enumerate(binary[::-1]):
            decimal += int(bit) * 2 ** i
        return decimal

    def bits2springs(self, bits):
        stack = []
        for bit in bits:
            if bit == '0':
                stack.append(UnitSpring())
            else:
                right = stack.pop()
                left = stack.pop() if stack else UnitSpring()
                stack.append(Spring.inParallel(left, right))
        while len(stack) > 1:
            right = stack.pop()
            left = stack.pop()
            stack.append(Spring.inSeries(left, right))
        return stack
    
    def compute_oscillations(self, bits: str, t: float, dt: float, x0: float, vo: float) -> List[float]:
        springs = self.bits2prings(bits)
        if len(springs) == 1:
            body = Spring(1.0, springs[0])
        else:
            left_springs = springs[:len(springs)//2]
            right_springs = springs[len(springs)//2:]
            left_body = Converter().bits_to_springs(left_springs)
            right_body = Converter().bits_to_springs(right_springs)
            body = Spring.inSeries(left_body, right_body)
        return body.move(t, dt, x0, vo)

    def calculate_frequency_amplitudes(self, bits: List[int]):
        springs = self.bits_to_springs(bits)
        oscillations = self.connect_body_to_springs(springs)
        n = len(oscillations)
        dt = 0.01
        freqs = np.fft.fftfreq(n, dt)[:n // 2]
        fft = np.fft.fft(oscillations)[:n // 2]
        amplitudes = 2.0 / n * np.abs(fft)
        return freqs, amplitudes

    def evaluate_decimal_value(self, bits: List[int]):
        freqs, amplitudes = self.calculate_frequency_amplitudes(bits)
        decimal_value = 0
        for i in range(len(amplitudes)):
            decimal_value += amplitudes[i] * freqs[i]
        return decimal_value



class ConverterInt(Converter):
    def __init__(self, spring_constant=1.0):
        self.spring_constant = spring_constant

    def bin2dec(self, binary):
        decimal = 0
        power = len(binary) - 1
        for bit in binary:
            decimal += int(bit) * (2 ** power)
            power -= 1
        return decimal

    def bits2prings(self, bits):
        springs = [UnitSpring(self.spring_constant) if bit == '0' else Spring(self.spring_constant) for bit in bits]
        spring = springs[0]
        for i in range(1, len(springs)):
            spring = spring.inSeries(springs[i])
        return [spring]

class ConverterFloat(Converter):

    def __init__(self, integer_bits, fraction_bits):
        super().__init__()
        self.integer_bits = integer_bits
        self.fraction_bits = fraction_bits
        self.integer_converter = ConverterInt()
        self.fraction_converter = ConverterInt()

    def bin2dec(self, binary):
        point = binary.find('.')
        if (point == -1):
            point = len(binary)

        dec = 0
        pw = 1

        for i in range(point - 1, -1, -1):
            dec += ((ord(binary[i]) -ord('0')) * pw)
            pw *= 2

        frac = 0
        pw = 2

        for i in range(point + 1, len(binary)):
            frac += ((ord(binary[i]) -ord('0')) / pw)
            pw *= 2.0

        return dec + frac

    def bits2springs(self, bits):
        integer = bits[:self.integer_bits]
        frac = bits[self.integer_bits:]

        intSprings = self.integer_converter.bits2springs(integer)
        fracSprings = self.fraction_converter.bits2springs(frac)

        # Connect the integer and fraction springs in parallel
        if len(intSprings) == 0:
            return fracSprings
        elif len(fracSprings) == 0:
            return intSprings
        else:
            return [Spring.inParallel(intSprings[-1], fracSprings[0])] + intSprings[:-1] + fracSprings[1:]

#Test of converters
converter = Converter8Bit()
binSequences = ['00000110', '00010001', '00111010', '00000100']
for seq in binSequences:
    decimal = converter.bin2Dec(seq)
    springs_8bit = converter.bits2springs(binary_seq)
    print(f"Decimal is: {binary_seq} = {decimal}")

conv = ConverterInt()
binSequences1 = ['000', '000001', '0011010', '1111100']
for seq in binSequences1:
    decimal = conv.bi2dec(seq)
    int2 = conv.bits2prings(seq)
    print(f"The decimal value of binary sequence {seq} is {decimal}")

conv_float = ConverterFloat(2,3)

binary = "100.01"
decimal = conv_float.bin2dec(binary)
springs1 = conv_float.bits2springs(binary)
print(f"Binary: {binary}")
print(f"Decimal: {decimal}")

binary = "0.01101"
decimal = conv_float.bin2dec(binary)
springs = conv_float.bits2springs(binary)
print(f"Binary: {binary}")
print(f"Decimal: {decimal}")

binary = "110.101"
decimal = conv_float.bin2Dec(binary)
springs5 = conv_float.bits2springs(binary)
print(f"Binary: {binary}")
print(f"Decimal: {decimal}")