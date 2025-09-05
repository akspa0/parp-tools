using GillijimProject.Next.Core.Domain;
using Xunit;

namespace GillijimProject.Next.Tests;

public class SmokeTests
{
    [Fact]
    public void CanInstantiateDomainTypes()
    {
        var wdt = new WdtAlpha("sample.wdt");
        var adt = new AdtAlpha("map_00_00.adt");
        var lk = new AdtLk("map_00_00");
        Assert.NotNull(wdt);
        Assert.NotNull(adt);
        Assert.NotNull(lk);
    }
}
