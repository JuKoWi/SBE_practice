{
    inputs = { 
        nixpkgs.url = "github:nixos/nixpkgs";
    };

    outputs = { self, nixpkgs }:
        let 
             pkgs = import nixpkgs {
                 system="x86_64-linux";
                 config.allowUnfree = true;
             };

             myPython = pkgs.python3.withPackages (p: with p; [
                numpy
                scipy
                matplotlib
             ]);

        in {
           devShell.x86_64-linux =
                pkgs.mkShell {
                    buildInputs = [
                           myPython
                           # pkgs.gtest
                    ];
                };
    };
}
