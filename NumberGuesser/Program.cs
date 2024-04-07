// See https://aka.ms/new-console-template for more information

Console.WriteLine("Hello, World!");

var network = new Neural.Network(2, 1, 3, 2);
foreach (var f in network.ForwardPass()) Console.WriteLine(f);

network.Train(
    new[] { 0.32f, 0.42f },
    new[] { 1.0f }
);
