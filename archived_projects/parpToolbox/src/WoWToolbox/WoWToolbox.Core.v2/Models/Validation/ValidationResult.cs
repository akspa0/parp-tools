using System.Collections.Generic;
using System.Linq;

namespace WoWToolbox.Core.v2.Models.Validation
{
    public class ValidationResult
    {
        public bool IsValid => !Errors.Any();
        public List<string> Errors { get; } = new List<string>();

        public void AddError(string error)
        {
            Errors.Add(error);
        }
    }
}
