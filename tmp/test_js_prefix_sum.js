// Simple test to verify the JS prefix sum implementation
// This will help us understand if the algorithm itself works with 513 elements

// CPU reference implementation
function prefixSumExclusive(input) {
    const result = [0];
    for (let i = 0; i < input.length - 1; i++) {
        result.push(result[result.length - 1] + input[i]);
    }
    return result;
}

// Test with 512 elements (should work)
console.log("Testing with 512 elements:");
const input512 = Array.from({length: 512}, (_, i) => i);
const expected512 = prefixSumExclusive(input512);
console.log(`Input: [${input512.slice(0, 5).join(', ')}...${input512.slice(-5).join(', ')}]`);
console.log(`Expected: [${expected512.slice(0, 5).join(', ')}...${expected512.slice(-5).join(', ')}]`);
console.log(`Last element: expected ${expected512[511]}, input was ${input512[511]}`);

// Test with 513 elements (this is where Metal fails)
console.log("\nTesting with 513 elements:");
const input513 = Array.from({length: 513}, (_, i) => i);
const expected513 = prefixSumExclusive(input513);
console.log(`Input: [${input513.slice(0, 5).join(', ')}...${input513.slice(-5).join(', ')}]`);
console.log(`Expected: [${expected513.slice(0, 5).join(', ')}...${expected513.slice(-5).join(', ')}]`);
console.log(`Last element: expected ${expected513[512]}, input was ${input513[512]}`);

// Let's also test some edge cases
console.log("\nTesting edge cases:");
[1, 2, 3, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025].forEach(size => {
    const input = Array.from({length: size}, (_, i) => i);
    const expected = prefixSumExclusive(input);
    console.log(`Size ${size}: last element = ${expected[size-1]} (sum of 0..${size-2})`);
});