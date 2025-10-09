{
  description = "Dev environment for Tosa Nikki-based RAG";

  inputs = {
    nixpkgs.url      = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = inputs@{ self, flake-parts, gloss-tools, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      # Expose outputs for all major platforms
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      
      perSystem = { pkgs, inputs', lib, system, ... }: {
        # Override the provider's default devShell while reusing it as a base
        devShells.default = lib.mkForce (pkgs.mkShell {
          # Add TeX/graphics stack and extra tools
          packages = with pkgs; [
            uv
          ];

          # Keep shell startup fast; do not rebuild font cache on every entry.
          shellHook = ''
            echo "Tosa Nikki RAG dev"
          '';
        });
      };
    };
}
