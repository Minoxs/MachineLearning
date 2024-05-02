using Neural;

Console.WriteLine("Hello, World!");

var network = new Network(2, 1, 32, 64, 64);
foreach (var f in network.ForwardPass([0.32f, 0.42f])) Console.WriteLine(f);

for (int i = 0; i < 1000; i++) {
    network.Train(
        new[] { 0.32f, 0.42f },
        new[] { 1.3f }
    );
}

foreach (var f in network.ForwardPass([0, 0])) Console.WriteLine(f);
Console.WriteLine(network.Output[0]);
