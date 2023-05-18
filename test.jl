println("Hello World")

a = 14
if a % 2 == 0
    println("even")
elseif a % 2 == 1
    println("odd")
end

function add(x, y)
    wa = x+y
    return wa
end

b = 2
c = 3
d = add(b, c)
println(d)

# comment
#= comment
comment
commment
=#

#you can type Greek letters like latex
π
α = 5.0

b^c
b*c

#and
if (a%2 == 0)&&(a > 0)
    println(a, "is positive even number")
end

#or
if (a%2==0)||(a%3 == 0)
    println(a)
end